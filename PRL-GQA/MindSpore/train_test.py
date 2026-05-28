from network import DoubleFusion
# import torch
# import torch.nn as nn
# import numpy as np
from data_loader import PointCloudDataset
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
# import datetime
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

class MyHinge(nn.Cell):
    def __init__(self,margin):
        super(MyHinge,self).__init__()
        self.margin = margin
    def construct(self, dist1, dist2, label):
        value = label*(dist1-dist2)-self.margin
        value[value<0] = 0
        return ops.mean(value)

class LossFunc(nn.Cell):
    def __init__(self,lamda,margin1,margin2):
        super(LossFunc,self).__init__()
        self.crossloss = nn.BCELoss(reduction='mean')
        self.lowhinge = nn.MarginRankingLoss(margin=margin1)
        self.highhinge = MyHinge(margin2)
        self.lamda = lamda


    # def forward(self,p,g):
    #     g = g.view(-1, 1)
    #     p = p.view(-1, 1)
    #     loss_Fidelity = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))
    #     #Hinge_loss = torch.max(0,margin-(x1-x2))
    #     return torch.mean(loss_Fidelity)
    def construct(self,dist1,dist2,out,label):
        loss_cross = self.crossloss(out,label)
        loss_low = self.lowhinge(dist1,dist2,label)
        loss_high = self.highhinge(dist1,dist2,label)
        loss_std = loss_cross + self.lamda*(loss_high + loss_low)
        return ops.mean(loss_std)

class MyWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn
       self.num = None

   def construct(self, x, y, label):
       dist1, dist2, out = self._backbone(x, y)
       self.num = correct_num(dist1, dist2) #num/B，反映了预测准确率
       return self._loss_fn(out, label)

   @property
   def backbone_network(self):
       return self._backbone

def train(PCDataset,train_net, loss_net, epoch,_f_loss=None):   # 使用时
    loss=0
    accs = Ratio()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_net.set_train()
    # train_net = train_net.set_grad(requires_grad=True) #这个求梯度时会报错
    # net.train()
    trainloader = ds.GeneratorDataset(PCDataset, column_names=["data1_tensor", "data2_tensor", "label_tensor"],
        num_parallel_workers=3, shuffle=True, python_multiprocessing=False)
    trainloader = trainloader.batch(2, drop_remainder=False) #T4=2 V100=2
    batch_num = trainloader.get_dataset_size()
    print("batch_num:",batch_num)
    trainloader = trainloader.create_dict_iterator()
    for idx, input in enumerate(trainloader):
        data1 = input['data1_tensor']
        data2 = input['data2_tensor']
        label = input['label_tensor']
        data1 = ops.transpose(data1, (0, 1, 3, 2))
        # data1 = torch.transpose(data1, -1, -2) # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = ops.transpose(data2, (0, 1, 3, 2))
        # data2 = torch.transpose(data2, -1, -2)
        loss_ = train_net(data1,data2, label)  # 输出量在GPU上
        # num = correct_num(dist1, dist2)
        # num =num.cpu()

        # loss_net = criterion(out, label)

        #hingloss
        #loss_net = criterion(dist1,dist2, label)

        #loss_net = criterion(dist1, dist2,out,label)

        loss+=loss_.asnumpy().item()
        accs.update(loss_net.num.asnumpy().item(),data1.shape[0])
        # optimizer.zero_grad()
        # loss_net.backward()
        # optimizer.step()
        if idx % 50 ==49:
            print("loss:" + str(loss/50))
            if _f_loss is not None:
                _f_loss.write("loss:" + str(loss/50)+"\n")
            loss=0
    # acc=test(train_loader,net,criterion,epoch)
    print('\ntrain set: epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch, 100. * accs.ratio))


def test(PCDataset, net, criterion,epoch,_f_acc=None,mode=1):
    loss = Ratio()
    accs = Ratio()
    accs_comandown=Ratio()
    accs_noise=Ratio()
    net.set_train(False)
    # net = net.set_grad(requires_grad=False)
    # net.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        # num = num.cpu()
        loss_net = criterion(out, label)

        # hingloss
        #loss_net = criterion(dist1, dist2, label)

        #loss_net = criterion(dist1, dist2, out, label)

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
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    print("int(os.getenv('DEVICE_ID', '0')):  ",int(os.getenv('DEVICE_ID', '0'))) #0
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #如果开启的话，会在本地生成额外的过程文件，开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False) #多进程可以使用共享内存

    train_Dataset = PointCloudDataset('./rank_pair_train.txt')
    test_Dataset = PointCloudDataset('./rank_pair_test.txt',True)
    lossfile_path = './loss.txt'
    accfile_path = './ave_acc.txt'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    f_acc = open(accfile_path, 'w', encoding='utf-8')
    # trainloader = DataLoader(train_Dataset, batch_size=4, num_workers=0, shuffle=True,
    #                          drop_last=False)
    # testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
    #                         drop_last=False)

    net = DoubleFusion()
    ckpt = '/userhome/PRL-GQA/MindSpore/best.ckpt'
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        param_not_load = ms.load_param_into_net(net, param_dict)
        print("param_not_load: ", param_not_load)
        print('best_acc from model:',param_dict['best_acc'].value().asnumpy().item())
    # net = net.to(device)
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss(reduction='mean')
    #criterion =LossFunc(0.8,0.1,0.9)
    # criterion = torch.nn.MarginRankingLoss(margin=0.2)
    # optimizer =optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)
    cur_lr = 0.00001
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=cur_lr)
    # optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    epochs = 20
    best_acc = 0
    best_epoch = 0
    loss_net = MyWithLossCell(net, criterion)
    train_net = nn.TrainOneStepCell(loss_net, optimizer, sens=1.0)
    for epoch in range(1, epochs + 1):
        train(train_Dataset, train_net, loss_net, epoch, f_loss)
        if epoch%2 == 0:
            cur_lr = cur_lr * 0.8
            print("cur_lr:",cur_lr)
            ops.assign(optimizer.learning_rate, ms.Tensor(cur_lr, ms.float32))
        # StepLR.step()
        # with torch.no_grad:
        acc = test(test_Dataset, net, criterion, epoch, f_acc)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            ms.save_checkpoint(net, 'best.ckpt', append_dict={'best_acc':best_acc})
            # torch.save(net.score_compute.state_dict(), 'new_params.pth')
    print('\nbest_epoch: {:d}, best Accuracy: {:.2f}%\n'.format(
        best_epoch, 100. * best_acc))
    print(best_acc, best_epoch)
    print("finish training")
    f_loss.close()
    f_acc.close()


if __name__ == '__main__':
    main()