import argparse
import tensorflow as tf
from data_list import ImageList
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import xlrd
from scipy.optimize import curve_fit
import pandas as pd
from model import ITModel

def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    dataFile = sorted(dataFile)
    return dataFile


def cal_SROCC(pred, target):
    _, _, pred = logistic_5_fitting_no_constraint(pred, target)
    plcc, _ = pearsonr(pred, target)
    srocc, _ = spearmanr(pred, target)
    krocc, _ = kendalltau(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return plcc, srocc, krocc, rmse


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted

def test(args, model, test_set, epoch):
    pred_all = np.array([])
    target_all = np.array([])
    path_all = np.array([])

    for idx, input in enumerate(test_set.dataset):
        data = input[0]
        target = input[1]
        path = input[2]
        # Label scale normalization to 0-1
        target = (target) / args.tlabelscale

        output = model(data)
        pred = tf.reshape(output, target.shape).numpy()
        target = target.numpy()
        pred_all = np.concatenate((pred_all, pred), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        path_all = np.concatenate((path_all, path.numpy()), axis=0)

    plcc, srocc, krocc, rmse = cal_SROCC(pred_all, target_all)

    print(' '.join([
        f"PLCC: {plcc:.6f},",
        f"SROCC: {srocc:.6f}, ",
        f"KROCC: {krocc:.6f}, ",
        f"RMSE: {rmse:.6f}, "
    ]))

    return plcc, srocc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IT-PCQA')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='input batch size for testing')
    parser.add_argument('--slabelscale', type=float, default=9.0,
                        help='Maximum value of labels for source domain')
    parser.add_argument('--tlabelscale', type=float, default=9.0,
                        help='Maximum value of labels for target domain')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--resume', type=str, default='best_23.h5',
                  help='path for loading the checkpoint')
    parser.add_argument('--test_list', type=str, default='/code/it-pcqa/config/SJTU-PCQA/label_yq0-9_val.txt',
                  help='path for val txt')

    args = parser.parse_args()

    # random
    if not os.path.exists('results'):
        os.makedirs('results')
    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # loading dataset (source domain for images, and target domain for point cloud projections)
    test_list = args.test_list

    # resize
    pic_resize = 224
    channel = 256

    # create DataLoader
    test_set = ImageList(test_list, args.test_batch_size, pic_resize, mode='RGB')

    model = ITModel(channel)
    # model.build(input_shape = (2, 3, 224, 224))
    img = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(8, 3, 224, 224))
    data_source = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    data_target = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    label_source = tf.keras.initializers.RandomUniform(minval=0, maxval=1.0)(shape=(4,))
    loss3, score = model(img,data_source,label_source,data_target,is_train=True) #初始化后要先跑一遍

    if args.resume != None and os.path.exists(args.resume):
        model.load_weights(args.resume)
    plcc, srocc = test(args, model, test_set, 0)

if __name__ == '__main__':
    main()
