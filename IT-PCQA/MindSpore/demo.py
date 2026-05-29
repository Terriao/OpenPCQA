import argparse
import mindspore as ms
import mindspore as mindspore
import mindspore.dataset as ds
from mindspore import context
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from data_list import ImageList
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import xlrd
from scipy.optimize import curve_fit
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

def test(args, model, test_loader, epoch):
    model.set_train(False)

    pred_all = np.array([])
    target_all = np.array([])
    path_all = np.array([])
    test_loader = test_loader.create_dict_iterator()

    for idx, input in enumerate(test_loader):
        data = input["img"]
        target = input["target"]
        path = input["path"]

        # Label scale normalization to 0-1
        target = (target) / args.tlabelscale

        output = model(data)
        pred = output.view(target.shape).asnumpy()
        target = target.asnumpy()
        pred_all = np.concatenate((pred_all, pred), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        path_all = np.concatenate((path_all, path.asnumpy()), axis=0)

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
    parser.add_argument('--resume', type=str, default='best_41.ckpt',
                  help='path for loading the checkpoint')
    parser.add_argument('--test_list', type=str, default='/code/it-pcqa/config/SJTU-PCQA/label_yq0-9_val.txt',
                  help='path for val txt')

    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')
    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    print("int(os.getenv('DEVICE_ID', '0')):  ",int(os.getenv('DEVICE_ID', '0'))) #0
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #如果开启的话，会在本地生成额外的过程文件，开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False) #多进程可以使用共享内存

    # loading dataset (source domain for images, and target domain for point cloud projections)
    test_list = args.test_list

    # resize
    pic_resize = 224
    channel = 256

    # create DataLoader
    test_set = ImageList(test_list, transform=transforms.Compose([
            vision.Resize([pic_resize, pic_resize], vision.Inter.BILINEAR),
            vision.ToTensor(),
            vision.Normalize((0.5,), (0.5,), is_hwc=False)
        ]), mode='RGB')
    test_loader = ds.GeneratorDataset(test_set, column_names=["img", "target", "path"],
        num_parallel_workers=2, shuffle=True, python_multiprocessing=False) #使用多线程
    test_loader = test_loader.batch(args.test_batch_size, drop_remainder=False)

    model = ITModel(channel)

    if args.resume != None and os.path.exists(args.resume):
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print("param_not_load: ", param_not_load)

    plcc, srocc = test(args, model, test_loader, 0)

if __name__ == '__main__':
    main()
