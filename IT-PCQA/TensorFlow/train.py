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

def train(args, model, source_list, target_list, pic_resize, trainset, trainset1, epoch, optimizer):
    len_source = len(trainset.dataset)
    len_target = len(trainset1.dataset)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    iter_source = iter(trainset.dataset)
    iter_target = iter(trainset1.dataset)
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0 and batch_idx > 0:
            trainset = ImageList(source_list, args.batch_size, pic_resize, mode='RGB')
            iter_source = iter(trainset.dataset)
        if batch_idx % len_target == 0 and batch_idx > 0:
            trainset1 = ImageList(target_list, args.batch_size, pic_resize, mode='RGB')
            iter_target = iter(trainset1.dataset)
        data_source, label_source, _ = next(iter_source)
        data_target, label_target,_ = next(iter_target)

        # Label scale normalization to 0-1
        label_source = (label_source) / args.slabelscale
        label_target = (label_target) / args.slabelscale

        with tf.GradientTape() as tape: #这里区别训练和推理，tf没有额外的set_train
            loss3, score = model(tf.concat([data_source, data_target], axis=0),data_source,label_source,data_target,True) #source是图片，target是点云
            loss1 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)\
                (label_source, score[:data_source.shape[0], 0])
            loss = loss1 + loss3
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if (batch_idx + epoch * num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, num_iter * args.batch_size,
                       100. * batch_idx / num_iter, loss.numpy().item(), loss1.numpy().item(), loss3.numpy().item()))

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

    path_all = path_all.reshape(-1, 1)
    target_all = target_all.reshape(-1, 1)
    pred_all = pred_all.reshape(-1, 1)
    all_results = np.concatenate((path_all, target_all, pred_all), axis=1)
    results2 = pd.DataFrame(columns=['plyname', 'MOS', 'pred'], data=all_results)
    results2.to_csv(f'results/test_pre_score{str(epoch)}.csv',index=False)

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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--slabelscale', type=float, default=9.0,
                        help='Maximum value of labels for source domain')
    parser.add_argument('--tlabelscale', type=float, default=9.0,
                        help='Maximum value of labels for target domain')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    parser.add_argument('--resume', type=str, default=None,
                  help='path for loading the checkpoint')
    parser.add_argument('--backbone', type=str, default='HSCNN',
                        help='backbone')

    args = parser.parse_args()

    # random
    if not args.random:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if not os.path.exists('results'):
        os.makedirs('results')
    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # loading dataset (source domain for images, and target domain for point cloud projections)
    root_path = '/userhome/IT-PCQA/'
    source_list = root_path + 'config/TID2013/mos_with_names.txt'
    target_list = root_path + 'config/SJTU-PCQA/label_yq0-9_train.txt'
    test_list = root_path + 'config/SJTU-PCQA/label_yq0-9_val.txt'

    # resize
    backbone = args.backbone
    if backbone == 'HSCNN':
        pic_resize = 224
        channel = 256

    # create DataLoader
    train_set = ImageList(source_list, args.batch_size, pic_resize, mode='RGB')    
    train_set1 = ImageList(target_list, args.batch_size, pic_resize, mode='RGB')
    test_set = ImageList(test_list, args.test_batch_size, pic_resize, mode='RGB')

    model = ITModel(channel,backbone=backbone)
    # model.build(input_shape = (2, 3, 224, 224))
    img = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(8, 3, 224, 224))
    data_source = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    data_target = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    label_source = tf.keras.initializers.RandomUniform(minval=0, maxval=1.0)(shape=(4,))
    loss3, score = model(img,data_source,label_source,data_target,is_train=True) #初始化后要先跑一遍

    if args.resume != None and os.path.exists(args.resume):
        model.load_weights(args.resume)

    # SGD optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
    result = []
    best_plcc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, source_list, target_list, pic_resize, train_set, train_set1, epoch, optimizer)
        plcc, srocc = test(args, model, test_set, epoch)
        if plcc > best_plcc:
            best_plcc = plcc
            best_epoch = epoch

            model.save_weights(f'best_{str(epoch)}.h5')
            print("save ckpt on epoch: ",epoch,", with best plcc: ",best_plcc)
        result.append([epoch, plcc, srocc])
        resultlist = pd.DataFrame(columns=['epoch','plcc', 'SROCC'], data=result)
        resultlist.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
