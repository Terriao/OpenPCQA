# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import pandas as pd
import mindspore as ms
import mindspore as mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
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
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    print("int(os.getenv('DEVICE_ID', '0')):  ",int(os.getenv('DEVICE_ID', '0'))) #0
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #如果开启的话，会在本地生成额外的过程文件，开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False) #多进程可以使用共享内存

    model = ResNet_mean_with_fast.resnet50(pretrained=False)
    # model.load_state_dict(torch.load(config.pretrained_model_path))
    param_dict = mindspore.load_checkpoint(config.pretrained_model_path)
    param_not_load = mindspore.load_param_into_net(model, param_dict)#加载可训练和不可训练的Parameter参数
    print('Load checkpoint from '+config.pretrained_model_path)
    print("param_not_load: ", param_not_load) #打印网络中没有被加载的参数，正常应该为空
    print('SROCC from model: ',param_dict['SROCC'].value().asnumpy().item())

    model.set_train(False)
    ## training data
    images_dir = config.path_imgs
    data_3d_dir = config.path_3d_features
    datainfo_test = config.data_info
    transformations_test = transforms.Compose([vision.CenterCrop(224),vision.ToTensor(),\
            vision.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], is_hwc=False)])
    testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, 
        transformations_test, crop_size=224)
    
    ## initialize dataloader
    n_test = len(testset)
    test_loader = ds.GeneratorDataset(testset, column_names=["imgs", "features", "labels", "video_name"],
        num_parallel_workers=config.num_workers, shuffle=False, python_multiprocessing=False)
    test_loader = test_loader.batch(1, drop_remainder=False)
    y_output = np.zeros(n_test)
    y_test = np.zeros(n_test)
    y_name = ['video_name'] * n_test

    # begin inference
    test_dataloader = test_loader.create_dict_iterator()
    for i, data in enumerate(test_dataloader): 
        video_name = data['video_name'].asnumpy().item()
        print(video_name)
        imgs = data['imgs']
        features = data['features']
        y_test[i] = data['labels'][0].asnumpy().item()
        outputs = model(imgs, features) #imgs.shape: (1, 4, 3, 224, 224)  features.shape: (1, 4, 256)
        y_output[i] = outputs[0].asnumpy().item()
        y_name[i] = video_name
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
        default='/userhome/VQA_PC-main/mindspore/train/ResNet_mean_with_fast_SJTU_7_best.ckpt', type=str)
    parser.add_argument('--path_imgs', default='/userhome/VQA_PC-main/train/database/sjtu_2d/', type=str)
    parser.add_argument('--path_3d_features', 
        default='/userhome/VQA_PC-main/train/database/sjtu_slowfast/', type=str)
    parser.add_argument('--data_info', default='/userhome/VQA_PC-main/test/data_info/sjtu_mos.csv', type=str)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--output_csv_path', type=str, default = 'prediction.csv')

    config = parser.parse_args()

    start_time = time.time()
    main(config)
    test_time = time.time() - start_time
    print('total test time: {:.3f}s'.format(test_time)) #total test time: 26.405s 3110MiB显存
