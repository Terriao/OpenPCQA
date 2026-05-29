# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def dataset_config(db_path,config,split,epoch):
    if config.database == 'SJTU':
        images_dir = db_path+'sjtu_2d/'
        datainfo_train = db_path+'sjtu_data_info/train_' + str(split+1) +'.csv'
        datainfo_test = db_path+'sjtu_data_info/test_' + str(split+1) +'.csv'
        data_3d_dir = db_path+'sjtu_slowfast/'
        trainset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_train, 
            crop_size=config.crop_size, epoch=epoch, batch_size=config.train_batch_size,is_train=True, 
            frame_index=config.frame_index, video_length_read = config.video_length_read)
        testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
            crop_size=config.crop_size, epoch=epoch, batch_size=1,is_train=False, frame_index=config.frame_index,
            video_length_read = config.video_length_read)
    elif config.database == 'WPC':
        images_dir = db_path+'wpc_2d/'
        datainfo_train = db_path+'wpc_data_info/train_' + str(split+1) +'.csv'
        datainfo_test = db_path+'wpc_data_info/test_' + str(split+1) +'.csv'
        data_3d_dir = db_path+'wpc_slowfast/'
        trainset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_train, 
            crop_size=config.crop_size, epoch=epoch, batch_size=config.train_batch_size,is_train=True, 
            frame_index=config.frame_index, video_length_read = config.video_length_read)
        testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
            crop_size=config.crop_size, epoch=epoch, batch_size=1,is_train=False, frame_index=config.frame_index,
            video_length_read = config.video_length_read)
    return trainset,testset

def main(config):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    first_flag = True
    best_all = np.zeros([config.split_num, 4])
    final_all = np.zeros([config.split_num, 4])
    db_path = '/userhome/VQA_PC-main/train/database/'
    for split0 in range(config.split_num): #0-8
        split = config.split_num-1-split0
        # model
        if config.model_name == 'ResNet_mean_with_fast':
            print('The current model is ' + config.model_name)
            model = ResNet_mean_with_fast.ResNet(pretrained=True, ckpt_path='/userhome/VQA_PC-main/tensorflow/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

        criterion = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) #对结果求mean

        param_num = 0
        for param in model.variables:
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6)) #23.86 million
        
        print('***********************************************************')
        print('Using '+ str(split+1) + '-th split.' )

        best_test_criterion = -1  # SROCC min
        
        print('Starting training:')
        for epoch in range(config.epochs):
            if epoch == 0:
                ## dataloader
                trainset,testset = dataset_config(db_path,config,split,epoch)
                n_test = len(testset.dataset)
                # optimizer
                batch_num = len(trainset.dataset) #返回一个epoch中的batch数 batchsize为8时返回42
                print('batch_num:',batch_num) #batchsize为16时返回21
                milestone = list(range(0,(config.epochs+1)*batch_num,config.decay_interval*batch_num))[1:] #milestone: [210, 420, 630]
                learning_rates = [config.conv_base_lr]
                for i in range(len(milestone)-1):
                    learning_rates.append(learning_rates[-1]*config.decay_ratio)
                learning_rates.append(learning_rates[-1])
                print("milestone:",milestone)
                print("learning_rates:",learning_rates)#[4e-05, 3.6e-05, 3.24e-05, 3.24e-05]
                learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(milestone, learning_rates)
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
            else:
                trainset,testset = dataset_config(db_path,config,split,epoch)
            batch_losses = []
            batch_losses_each_disp = []
            for i, (video, features, labels, video_name) in enumerate(trainset.dataset):
                with tf.GradientTape() as tape: #这里区别训练和推理，tf没有额外的set_train
                    outputs= model(video, features)
                    loss = criterion(labels, outputs)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                batch_losses.append(loss.numpy())
                batch_losses_each_disp.append(loss.numpy())
            avg_loss = sum(batch_losses) / batch_num
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            steps = optimizer.iterations
            print("steps:",steps.numpy())
            lr = learning_rate_fn(steps)
            print('The current learning rate is {:.06f}'.format(lr))

            # Test
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)
            # do validation after each epoch
            for i, (video, features, labels, video_name) in enumerate(testset.dataset):
                y_test[i] = labels[0].numpy().item()
                outputs = model(video, features, training_flag=False)
                y_output[i] = outputs[0].numpy().item()
        
            y_output_logistic = fit_function(y_test, y_output)
            test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
            test_SROCC = stats.spearmanr(y_output, y_test)[0]
            test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
            test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
            print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
            final = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
            final_all[split, :] = final
            
            final_res = {}
            final_res['type'] = 'final'
            final_res['epoch'] = epoch+1
            final_res['split_num'] = split+1
            final_res['SROCC'] = final[0]
            final_res['KROCC'] = final[1]
            final_res['PLCC'] = final[2]
            final_res['RMSE'] = final[3]
            save_result = pd.DataFrame([final_res]).copy(deep=True)
            if test_SROCC > best_test_criterion:
                print("Update best model using best_val_criterion ")
                ckpt = config.ckpt_path + '/' + config.model_name +'_' + config.database +'_' + str(split+1) + '_' + 'best.h5'
                model.save_weights(ckpt)
                # scio.savemat(trained_model_file+'.mat',{'y_pred':y_pred,'y_test':y_test})
                best = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                best_test_criterion = test_SROCC  # update best val SROCC
                best_all[split, :] = best
                print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
                best_res = {}
                best_res['type'] = 'best'
                best_res['epoch'] = epoch+1
                best_res['split_num'] = split+1
                best_res['SROCC'] = best[0]
                best_res['KROCC'] = best[1]
                best_res['PLCC'] = best[2]
                best_res['RMSE'] = best[3]
            save_result = save_result.append(best_res, ignore_index=True)
            if first_flag:
                save_result.to_csv(config.ckpt_path + '/' + 'save_results.csv', index=False)
            else:
                tmp_save_result = all_result.append(save_result)
                tmp_save_result.to_csv(config.ckpt_path + '/' + 'save_results.csv', index=False)
            print()
        if first_flag:
            all_result = save_result.copy(deep=True)
            first_flag = False
        else:
            all_result = all_result.append(save_result)
        print('Training completed.')
        print('*************************************************************************************************************************')
    final_median = np.median(final_all, 0)
    best_median = np.median(best_all, 0)
    print('*************************************************************************************************************************')
    print("The final median val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(final_median[0], final_median[1], final_median[2], final_median[3]))
    print("The best median val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_median[0], best_median[1], best_median[2], best_median[3]))
    print('*************************************************************************************************************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', default='SJTU', type=str)
    parser.add_argument('--model_name', default='ResNet_mean_with_fast', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=0.00004)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    # parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--split_num', type=int, default=9)#设置成9，每个都测一遍，然后求平均值作为性能结果，注意不是10，最后一个数据不是SJTU的
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--frame_index', type=int, default=5)
    parser.add_argument('--video_length_read', type=int, default=4)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/userhome/VQA_PC-main/tensorflow/train/SJTU_ckpts/')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)

    config = parser.parse_args()

    main(config)
