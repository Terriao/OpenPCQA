from network import DoubleFusion
from data_loader import PointCloudDataset
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
import time
import mindspore.ops as ops
import os

esp = 1e-8

def correct_num(dista, distb):    # 该函数可以换成计算out值>0.5的概率
    margin = 0
    pred = dista - distb - margin
    return (pred > 0).sum()*1.0

class Ratio(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.num1=0
        self.num2=0
        self.ratio=0
    def update(self,num1,num2):
        self.num1+=num1
        self.num2+=num2
        self.ratio=self.num1/self.num2

def test(PCDataset, net, criterion,_f_acc=None,mode=1):
    loss = Ratio()
    accs = Ratio()
    accs_comandown=Ratio()
    accs_noise=Ratio()
    net.set_train(False)
    # net = net.set_grad(requires_grad=False)
    test_loader = ds.GeneratorDataset(PCDataset, column_names=["data1_tensor", "data2_tensor", "label_tensor", "category"],
        num_parallel_workers=2, shuffle=False, python_multiprocessing=False)
    test_loader = test_loader.batch(1, drop_remainder=False)
    batch_num = test_loader.get_dataset_size()
    print("batch_num:",batch_num) #12600
    test_loader = test_loader.create_dict_iterator()
    for idx, input in enumerate(test_loader):
        # for idx, (data1, data2, label,category) in enumerate(test_loader):
        data1 = input['data1_tensor']
        data2 = input['data2_tensor']
        label = input['label_tensor']
        category = input['category']
        data1 = ops.transpose(data1, (0, 1, 3, 2))
        # data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = ops.transpose(data2, (0, 1, 3, 2))
        # data2 = torch.transpose(data2, -1, -2)
        dist1, dist2, out = net(data1,data2)
        num = correct_num(dist1, dist2)
        loss_net = criterion(out, label)

        loss.update(loss_net.asnumpy().item()*data1.shape[0], data1.shape[0])
        accs.update(num.asnumpy().item(), data1.shape[0])
        if category.asnumpy().item() == 'com&down':
            accs_comandown.update(num.asnumpy().item(),data1.shape[0])
        else:
            accs_noise.update(num.asnumpy().item(),data1.shape[0])
    if mode==1:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss.ratio, 100. * accs.ratio))
        print('com & down Accuracy: {:.4f}%, noise Accuracy: {:.4f}%\n'.format(
            100. * accs_comandown.ratio, 100. * accs_noise.ratio))
    else:
        print('\ntrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            loss.ratio, 100. * accs.ratio))
    if _f_acc is not None:
        _f_acc.write('Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss.ratio, 100. * accs.ratio))
    return accs.ratio

def main():
    start_time = time.time()
    print("start time:",start_time)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    print("int(os.getenv('DEVICE_ID', '0')):  ",int(os.getenv('DEVICE_ID', '0'))) #0
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #如果开启的话，会在本地生成额外的过程文件，开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False) #多进程可以使用共享内存

    test_Dataset = PointCloudDataset('./rank_pair_test.txt',True)
    lossfile_path = './loss.txt'
    accfile_path = './ave_acc.txt'
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    f_acc = open(accfile_path, 'w', encoding='utf-8')

    net = DoubleFusion()
    ckpt = '/userhome/PRL-GQA/MindSpore/best.ckpt'
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        param_not_load = ms.load_param_into_net(net, param_dict)
        print("param_not_load: ", param_not_load)
        print('best_acc from model:',param_dict['best_acc'].value().asnumpy().item())
    criterion = nn.BCELoss(reduction='mean')
    acc = test(test_Dataset, net, criterion, f_acc)
    print("acc:",acc)
    f_loss.close()
    f_acc.close()
    end_time = time.time()
    print("test time(s):",end_time - start_time)

if __name__ == '__main__':
    main()