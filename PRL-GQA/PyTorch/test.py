from network import double_fusion
import torch
import torch.nn as nn
import numpy as np
from data_loader import PointCloudDataset
from torch.utils.data import Dataset, DataLoader
esp = 1e-8
import time

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

def test(test_loader, net, criterion,_f_acc=None,mode=1):
    loss = Ratio()
    accs = Ratio()
    accs_comandown=Ratio()
    accs_noise=Ratio()
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for idx, (data1, data2, label,category) in enumerate(test_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        label = label.to(device)
        data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = torch.transpose(data2, -1, -2)
        dist1, dist2, out = net(data1,data2)
        num = correct_num(dist1, dist2)
        num = num.cpu()
        loss_net = criterion(out, label)

        loss.update(loss_net.cpu().detach().item()*data1.size()[0], data1.size()[0])
        accs.update(num, data1.size()[0])
        if category[0] == 'com&down':
            accs_comandown.update(num,data1.size()[0])
        else:
            accs_noise.update(num,data1.size()[0])
    if mode==1:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss.ratio, 100. * accs.ratio))
        print('com & down Accuracy: {:.4f}%, noise Accuracy: {:.4f}%\n'.format(
            100. * accs_comandown.ratio, 100. * accs_noise.ratio))
    else:
        print('\ntrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            loss.ratio, 100. * accs.ratio))
    if _f_acc is not None:
        _f_acc.write(str(accs.ratio.item())+'\n')
    return accs.ratio

def main():
    start_time = time.time()
    print("start time:",start_time)
    test_Dataset = PointCloudDataset('./rank_pair_test.txt',True)
    lossfile_path = './loss.txt'
    accfile_path = './ave_acc.txt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    f_acc = open(accfile_path, 'w', encoding='utf-8')
    testloader = DataLoader(test_Dataset, batch_size=1, num_workers=2, shuffle=False,
                            drop_last=False)

    net = double_fusion()
    net = net.to(device)
    criterion = nn.BCELoss()
    acc = test(testloader, net, criterion, f_acc)
    print("acc:",acc)
    f_loss.close()
    f_acc.close()
    end_time = time.time()
    print("test time(s):",end_time - start_time)


if __name__ == '__main__':
    main()