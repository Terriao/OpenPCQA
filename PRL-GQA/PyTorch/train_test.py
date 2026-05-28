from network import double_fusion
import torch
import torch.nn as nn
import numpy as np
from data_loader import PointCloudDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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



class myhinge(nn.Module):
    def __init__(self,margin):
        super(myhinge,self).__init__()
        self.margin = margin
    def forward(self, dist1, dist2, label):
        value = label*(dist1-dist2)-self.margin
        value[value<0] = 0
        return torch.mean(value)

class LossFunc(nn.Module):
    def __init__(self,lamda,margin1,margin2):
        super(LossFunc,self).__init__()
        self.crossloss = nn.BCELoss()
        self.lowhinge = nn.MarginRankingLoss(margin=margin1)
        self.highhinge = myhinge(margin2)
        self.lamda = lamda


    # def forward(self,p,g):
    #     g = g.view(-1, 1)
    #     p = p.view(-1, 1)
    #     loss_Fidelity = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))
    #     #Hinge_loss = torch.max(0,margin-(x1-x2))
    #     return torch.mean(loss_Fidelity)
    def forward(self,dist1,dist2,out,label):
        loss_cross = self.crossloss(out,label)
        loss_low = self.lowhinge(dist1,dist2,label)
        loss_high = self.highhinge(dist1,dist2,label)
        loss_std = loss_cross + self.lamda*(loss_high + loss_low)
        return torch.mean(loss_std)
def train(train_loader,net, criterion, optimizer,epoch,_f_loss=None):   # 使用时
    loss=0
    accs = Ratio()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.train()
    for idx, (data1, data2, label) in enumerate(train_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        label = label.to(device)
        data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = torch.transpose(data2, -1, -2)
        dist1, dist2, out = net(data1,data2)     # 输出量在GPU上
        num = correct_num(dist1, dist2)
        num =num.cpu()

        loss_net = criterion(out, label)

        #hingloss
        #loss_net = criterion(dist1,dist2, label)

        #loss_net = criterion(dist1, dist2,out,label)




        loss+=loss_net.cpu().detach().item()
        accs.update(num,data1.size()[0])
        optimizer.zero_grad()
        loss_net.backward()
        optimizer.step()
        if idx % 50 ==49:
            print("loss:" + str(loss/50))
            if _f_loss is not None:
                _f_loss.write(str(loss/50)+"\n")
            loss=0
    # acc=test(train_loader,net,criterion,epoch)
    print('\ntrain set: epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch, 100. * accs.ratio))


def test(test_loader, net, criterion,epoch,_f_acc=None,mode=1):
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

        # hingloss
        #loss_net = criterion(dist1, dist2, label)

        #loss_net = criterion(dist1, dist2, out, label)

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
    train_Dataset = PointCloudDataset('./rank_pair_train.txt')
    test_Dataset = PointCloudDataset('./rank_pair_test.txt',True)
    lossfile_path = './loss.txt'
    accfile_path = './ave_acc.txt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    f_acc = open(accfile_path, 'w', encoding='utf-8')
    trainloader = DataLoader(train_Dataset, batch_size=4, num_workers=0, shuffle=True,
                             drop_last=False)
    testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                            drop_last=False)

    net = double_fusion()
    net = net.to(device)
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    #criterion =LossFunc(0.8,0.1,0.9)
    # criterion = torch.nn.MarginRankingLoss(margin=0.2)
    # optimizer =optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    epochs = 20
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train(trainloader, net, criterion, optimizer, epoch, f_loss)
        StepLR.step()
        # with torch.no_grad:
        acc = test(testloader, net, criterion, epoch, f_acc)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(net.score_compute.state_dict(), 'new_params.pth')
    print('\nbest_epoch: {:d}, best Accuracy: {:.2f}%\n'.format(
        best_epoch, 100. * best_acc))
    print(best_acc, best_epoch)
    print("finish training")
    f_loss.close()
    f_acc.close()


if __name__ == '__main__':
    main()