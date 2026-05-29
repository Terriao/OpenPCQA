# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import pandas as pd
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


def main(config):
    # Load pre-trained model
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = ResNet_mean_with_fast.ResNet(pretrained=False)   
    input_data = tf.random.uniform(shape=[8,4, 224, 224, 3],minval=0,maxval=1.0,dtype=tf.float32)
    features = tf.random.uniform(shape=[8,4,256],minval=0,maxval=2.0,dtype=tf.float32)
    print(model(input_data,features))#要先call一遍，才能load_weights
    model.load_weights(config.pretrained_model_path)
    ## training data
    images_dir = config.path_imgs
    data_3d_dir = config.path_3d_features
    datainfo_test = config.data_info
    # transformations_test = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
    #     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    # testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, transformations_test, crop_size=224)
    testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
        crop_size=224, epoch=1, batch_size=1,is_train=False)
    
    ## initialize dataloader
    n_test = len(testset.dataset)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
    #     shuffle=False, num_workers=config.num_workers)
    y_output = np.zeros(n_test)
    y_test = np.zeros(n_test)
    y_name = ['video_name'] * n_test

    # begin inference
    for i, (imgs, features, labels, video_name) in enumerate(testset.dataset):
        print(video_name.numpy().item())
        y_test[i] = labels[0].numpy().item()
        outputs = model(imgs, features, training_flag=False)
        y_output[i] = outputs[0].numpy().item()
        y_name[i] =  video_name.numpy().item()
    y_output_logistic = fit_function(y_test, y_output)
    test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
    test_SROCC = stats.spearmanr(y_output, y_test)[0]
    test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
    test_KROCC = stats.stats.kendalltau(y_output, y_test)[0]
    print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))

    data = pd.DataFrame({'vid_name':y_name,'predicted_mos':y_output_logistic})
    data.to_csv(config.output_csv_path, index = None)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--pretrained_model_path', 
        default='/userhome/VQA_PC-main/tensorflow/train/SJTU_ckpts_调整BN/ResNet_mean_with_fast_SJTU_1_best.h5', type=str)
    parser.add_argument('--path_imgs', default='/userhome/VQA_PC-main/train/database/sjtu_2d/', type=str)
    parser.add_argument('--path_3d_features', 
        default='/userhome/VQA_PC-main/train/database/sjtu_slowfast/', type=str)
    parser.add_argument('--data_info', default='/userhome/VQA_PC-main/test/data_info/sjtu_mos.csv', type=str)
    # parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--output_csv_path', type=str, default = 'prediction.csv')

    config = parser.parse_args()

    start_time = time.time()
    main(config)
    test_time = time.time() - start_time
    print('total test time: {:.3f}s'.format(test_time)) #total test time: 44.887s，测试第2次的值
