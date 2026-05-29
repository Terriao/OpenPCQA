import argparse
import mindspore as ms
import mindspore as mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
import mindspore.ops as ops
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
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

class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.loss1 = 0.0
        self.loss3 = 0.0

    def construct(self, s_img1,data_source,label_source,data_target,is_train=True):
        loss3, score = self._backbone(s_img1,data_source,label_source,data_target,is_train)
        loss1 = self._loss_fn(score.narrow(0, 0, data_source.shape[0])[:,0], ops.Cast()(label_source, ms.float32))

        loss = loss1 + loss3
        self.loss1 = loss1.asnumpy().item()
        self.loss3 = loss3.asnumpy().item()
        return loss

    @property
    def backbone_network(self):
        return self._backbone

def train(args, train_loader, train_loader1, optimizer, epoch, model):
    model.set_train()

    len_source = train_loader.get_dataset_size()
    len_target = train_loader1.get_dataset_size()
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    def forward_fn(s_img1,data_source,label_source,data_target,is_train=True): #前向传播并计算loss
        loss3, score = model(s_img1,data_source,label_source,data_target,is_train)
        loss1 = nn.MSELoss()(score.narrow(0, 0, data_source.shape[0])[:,0], ops.Cast()(label_source, ms.float32))
        loss = loss1 + loss3
        loss1_ = loss1.asnumpy().item()
        loss3_ = loss3.asnumpy().item()
        return loss,loss1_,loss3_ #结合下面has_aux=True，只对第一项计算梯度

    grad_fn = mindspore.value_and_grad(forward_fn, #梯度函数
        grad_position=None, #只对网络变量求导
        weights=model.trainable_params(), #需要返回梯度的网络变量
        has_aux=True) #是否返回辅助参数的标志，若为True，fn输出数量必须超过一个，其中只有fn第一个输出参与求导，其他输出值将直接返回

    def train_step(s_img1,data_source,label_source,data_target,is_train=True): #一个batch step的操作
        (loss,loss1_,loss3_), grads = grad_fn(s_img1,data_source,label_source,data_target,is_train) #这里的输入参数直接传给forward_fn，返回的第一项是forward_fn的输出，第二项是梯度
        loss = ops.Depend()(loss, optimizer[3](grads[-4:])) #optimizer(grads)优化器更新参数，Depend确保梯度更新完成，再输出out_criterion
        optimizer[2](grads[-8:-4])
        optimizer[1](grads[-12:-8]) #grads的索引可以打印grads每一项的shape，与model.trainable_params()打印出来的顺序对应
        optimizer[0](grads[:-12]) #注意optimizer是3210，倒序
        return loss,loss1_,loss3_


    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = train_loader.create_dict_iterator()
        if batch_idx % len_target == 0:
            iter_target = train_loader1.create_dict_iterator()
        source = next(iter_source)
        data_source, label_source,_ = source["img"],source["target"],source["path"]
        target = next(iter_target)
        data_target, label_target,_ = target["img"],target["target"],target["path"]

        # Label scale normalization to 0-1
        label_source = (label_source) / args.slabelscale
        label_target = (label_target) / args.slabelscale

        loss,loss1_,loss3_ = train_step(ops.concat((data_source, data_target), 0),data_source,label_source,data_target,True)

        if (batch_idx + epoch * num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, num_iter * args.batch_size,
                       100. * batch_idx / num_iter, loss.asnumpy().item(), loss1_, loss3_))

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
        mindspore.set_seed(args.seed)
        np.random.seed(args.seed)
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
    train_set = ImageList(source_list, transform=transforms.Compose([
            vision.Resize([pic_resize, pic_resize], vision.Inter.BILINEAR),
            vision.ToTensor(), #shape将从(H, W, C)调整为(C, H, W)默认输出数据类型为numpy.float32
            vision.Normalize((0.5,), (0.5,), is_hwc=False)
        ]), mode='RGB')
    train_loader = ds.GeneratorDataset(train_set, column_names=["img", "target", "path"],
        num_parallel_workers=3, shuffle=True, python_multiprocessing=False) #使用多线程
    train_loader = train_loader.batch(args.batch_size, drop_remainder=False)
    
    train_set1 = ImageList(target_list, transform=transforms.Compose([
                vision.Resize([pic_resize, pic_resize], vision.Inter.BILINEAR),
                vision.ToTensor(),
                vision.Normalize((0.5,), (0.5,), is_hwc=False)
            ]), mode='RGB')
    train_loader1 = ds.GeneratorDataset(train_set1, column_names=["img", "target", "path"],
        num_parallel_workers=3, shuffle=True, python_multiprocessing=False) #使用多线程
    train_loader1 = train_loader1.batch(args.batch_size, drop_remainder=False)

    test_set = ImageList(test_list, transform=transforms.Compose([
            vision.Resize([pic_resize, pic_resize], vision.Inter.BILINEAR),
            vision.ToTensor(),
            vision.Normalize((0.5,), (0.5,), is_hwc=False)
        ]), mode='RGB')
    test_loader = ds.GeneratorDataset(test_set, column_names=["img", "target", "path"],
        num_parallel_workers=2, shuffle=True, python_multiprocessing=False) #使用多线程
    test_loader = test_loader.batch(args.test_batch_size, drop_remainder=False)

    model = ITModel(channel,backbone=backbone)

    if args.resume != None and os.path.exists(args.resume):
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print("param_not_load: ", param_not_load)

    # SGD optimizer
    optimizer_model = nn.SGD(params=model.extraction.trainable_params(), learning_rate=args.lr, momentum=args.momentum, 
        weight_decay=0.0005)
    optimizer_mapping = nn.SGD(params=model.mapping.trainable_params(), learning_rate=args.lr, momentum=args.momentum, 
        weight_decay=0.0005)
    optimizer_regression = nn.SGD(params=model.regression.trainable_params(), learning_rate=args.lr, momentum=args.momentum, 
        weight_decay=0.0005)
    optimizer_adnet = nn.SGD(params=model.adnet.trainable_params(), learning_rate=args.lr, momentum=args.momentum, 
        weight_decay=0.0005)
    optimizer = [optimizer_model, optimizer_mapping, optimizer_regression, optimizer_adnet]
    result = []
    # loss_net = MyWithLossCell(model, nn.MSELoss())
    # train_net = nn.TrainOneStepCell(loss_net, optimizer, sens=1.0)
    best_plcc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, train_loader1, optimizer, epoch, model)
        plcc, srocc = test(args, model, test_loader, epoch)
        if plcc > best_plcc:
            best_plcc = plcc
            best_epoch = epoch
            ms.save_checkpoint(model, f'best_{str(epoch)}.ckpt', append_dict={'best_plcc':best_plcc})
            print("save ckpt on epoch: ",epoch,", with best plcc: ",best_plcc)
        result.append([plcc, srocc])
        resultlist = pd.DataFrame(columns=['plcc', 'SROCC'], data=result)
        resultlist.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
