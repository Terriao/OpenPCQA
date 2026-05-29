# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore as mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import time

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

class MyWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn

   def construct(self, x, y, label):
       out = self._backbone(x, y)
       return self._loss_fn(out, label)

   @property
   def backbone_network(self):
       return self._backbone

def main(config):
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    print("int(os.getenv('DEVICE_ID', '0')):  ",int(os.getenv('DEVICE_ID', '0'))) #0
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #如果开启的话，会在本地生成额外的过程文件，开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False) #多进程可以使用共享内存

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    first_flag = True
    best_all = np.zeros([config.split_num, 4])
    final_all = np.zeros([config.split_num, 4])
    for split0 in range(config.split_num): #0-8
        split = config.split_num-1-split0
        # model
        if config.model_name == 'ResNet_mean_with_fast':
            print('The current model is ' + config.model_name)
            model = ResNet_mean_with_fast.resnet50(pretrained=True, ckpt_path='/userhome/VQA_PC-main/resnet50_ascend_v190_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt')

        criterion = nn.MSELoss()

        param_num = 0
        for param in model.get_parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6)) #23.86 million
        
        print('***********************************************************')
        print('Using '+ str(split+1) + '-th split.' )

        transformations_train = transforms.Compose([vision.RandomCrop(224),vision.ToTensor(),\
                vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)])
        transformations_test = transforms.Compose([vision.CenterCrop(224),vision.ToTensor(),\
                vision.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], is_hwc=False)])
        db_path = '/userhome/VQA_PC-main/train/database/'
        if config.database == 'SJTU':
            images_dir = db_path+'sjtu_2d/'
            datainfo_train = db_path+'sjtu_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = db_path+'sjtu_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = db_path+'sjtu_slowfast/'
            trainset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_train, 
                transformations_train, crop_size=config.crop_size,frame_index=config.frame_index,
                video_length_read = config.video_length_read)
            testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
                transformations_test, crop_size=config.crop_size,frame_index=config.frame_index,
                video_length_read = config.video_length_read)
        elif config.database == 'WPC':
            images_dir = db_path+'wpc_2d/'
            datainfo_train = db_path+'wpc_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = db_path+'wpc_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = db_path+'wpc_slowfast/'
            trainset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_train, 
                transformations_train, crop_size=config.crop_size,frame_index=config.frame_index,
                video_length_read = config.video_length_read)
            testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
                transformations_test, crop_size=config.crop_size,frame_index=config.frame_index,
                video_length_read = config.video_length_read)
        
        ## dataloader
        train_loader = ds.GeneratorDataset(trainset, column_names=["video", "features", "labels", "_"],
            num_parallel_workers=config.num_workers, shuffle=True, python_multiprocessing=False) #使用多线程
        train_loader = train_loader.batch(config.train_batch_size, drop_remainder=False)

        test_loader = ds.GeneratorDataset(testset, column_names=["video", "features", "labels", "_"],
            num_parallel_workers=config.num_workers, shuffle=False, python_multiprocessing=False)
        test_loader = test_loader.batch(1, drop_remainder=False)

        best_test_criterion = -1  # SROCC min
        n_train = len(trainset) #336
        n_test = len(testset)

        # optimizer
        batch_num = train_loader.get_dataset_size() #返回一个epoch中的batch数 batchsize为8时返回42
        print('batch_num:',batch_num) #batchsize为16时返回21
        milestone = list(range(0,(config.epochs+1)*batch_num,config.decay_interval*batch_num))[1:]
        learning_rates = [config.conv_base_lr]
        for i in range(len(milestone)-1):
            learning_rates.append(learning_rates[-1]*config.decay_ratio)
        learning_rates.append(learning_rates[-1])
        milestone.append(milestone[-1]+batch_num*2) #使得最后一个epoch的lr不会为0
        lr = nn.piecewise_constant_lr(milestone, learning_rates)
        print('lr list len:',len(lr)) #batchsize为8时返回1302
        optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr, weight_decay=0.0000001)

        # loss_net = nn.WithLossCell(model, criterion) #接受数据和标签作为输入，并将返回loss
        loss_net = MyWithLossCell(model, criterion) #这个可以接受多个输入
        train_net = nn.TrainOneStepCell(loss_net, optimizer, sens=1.0)
        print('Starting training:')
        for epoch in range(config.epochs):
            train_net.set_train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            train_dataloader = train_loader.create_dict_iterator()
            for i, data in enumerate(train_dataloader):
                video = data['video']
                features = data['features']
                labels = data['labels']
                loss = train_net(video, features, labels).asnumpy()
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
            avg_loss = sum(batch_losses) / (n_train // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            lr = optimizer.get_lr()
            print('The current learning rate is {:.06f}'.format(lr.asnumpy()))

            # Test
            model.set_train(False)
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)
            # do validation after each epoch
            test_dataloader = test_loader.create_dict_iterator()
            for i, data in enumerate(test_dataloader):
                video = data['video']
                features = data['features']
                labels = data['labels']
                y_test[i] = labels[0].asnumpy().item()
                outputs = model(video, features)
                y_output[i] = outputs[0].asnumpy().item()
        
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
                mindspore.save_checkpoint(model, 
                    config.ckpt_path + '/' + config.model_name +'_' + config.database +'_' + str(split+1) + '_' + 'best.ckpt',
                    append_dict={'SROCC':test_SROCC})
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
        # mindspore.ms_memory_recycle()#回收内存，用这个指令会造成程序卡住
    final_median = np.median(final_all, 0)
    best_median = np.median(best_all, 0)
    print('*************************************************************************************************************************')
    print("The final median val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(final_median[0], final_median[1], final_median[2], final_median[3]))
    print("The best median val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_median[0], best_median[1], best_median[2], best_median[3]))
    print('*************************************************************************************************************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', default='WPC', type=str)
    parser.add_argument('--model_name', default='ResNet_mean_with_fast', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=0.00005)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--split_num', type=int, default=5)#设置成5，每个都测一遍，然后求平均值作为性能结果
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--frame_index', type=int, default=5)
    parser.add_argument('--video_length_read', type=int, default=4)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/userhome/VQA_PC-main/mindspore/train/ckpts')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)

    config = parser.parse_args()

    main(config)
