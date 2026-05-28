import torch
import torch.nn as nn
from pointnet_utils import PointNetEncoder
import torch.nn.functional as F
import numpy  as np
esp = 1e-8
class BasicFCModule(nn.Module):
    def __init__(self, inp_len=1024, oup_len=1):
        super(BasicFCModule, self).__init__()
        self.MLPLayers1 = nn.Sequential(
            nn.Linear(in_features=inp_len, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
        :param x:   N x C
        :return:    N x 1
        '''
        x = self.MLPLayers1(x)
        return x

# class my_pointnet(nn.Module):
#     def __init__(self,channel=3):
#         super(my_pointnet, self).__init__()
#         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#     def forward(self,x):
#         '''

#         :param x: B x D x number
#         :return:
#         '''
#         B, D, N = x.size()
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]  # B x 1024 X 1
#         x = x.view(-1, 1024)  # B x 1024
#         return x

class my_pointnet_deep(nn.Module):
    def __init__(self,channel=3):
        super(my_pointnet_deep, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
    def forward(self,x):
        '''

        :param x: B x D x number
        :return:
        '''
        B, D, N = x.size()
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = self.bn4(self.conv4(x3))
        x1 = torch.max(x1, 2, keepdim=True)[0]  # B x 64 X 1
        x1 = x1.view(-1, 64)
        x2 = torch.max(x2, 2, keepdim=True)[0]  # B x 128 X 1
        x2 = x2.view(-1, 128)
        x3 = torch.max(x3, 2, keepdim=True)[0]  # B x 256 X 1
        x3 = x3.view(-1, 256)
        x4 = torch.max(x4, 2, keepdim=True)[0]  # B x 512 X 1
        x4 = x4.view(-1, 512)

        x=torch.cat((x1,x2),-1)
        x = torch.cat((x, x3), -1)
        x = torch.cat((x, x4), -1)   # B x 960


        #x = torch.cat((x3, x4), -1)   # B x 960
        return x

class weight_score(nn.Module):
    def __init__(self):
        super(weight_score, self).__init__()
        # self.feature=PointNetEncoder(global_feat=True,feature_transform=True)
        # self.feature =my_pointnet()
        # self.score_mlp=BasicFCModule(1024)
        # self.weight_mlp=BasicFCModule(1024)
        self.feature = my_pointnet_deep()
        self.score_mlp = BasicFCModule(960)
        self.weight_mlp = BasicFCModule(960)
    def forward(self,x):
        '''
        :param x: B x patch_number x D x patch_size  // batch , patch, xyz, point_num
        :return: B x 1 //return batch*score
        '''
        B,patch_number,D,patch_size = x.size()
        x = x.view(-1,D,patch_size)    # pointnet网络输入只能接受3维    (B x patch_number) x D x patch_size
        fea_vector= self.feature(x)    # (B x patch_number) x 1024
        score = self.score_mlp(fea_vector)   # (B x patch_number) x 1
        weight = self.weight_mlp(fea_vector)  # (B x patch_number) x 1  线性层只能接受2维
        score = score.view(B,patch_number)
        weight = weight.view(B,patch_number)   # B x patch_number
        product_val = torch.mul(score,weight)
        product_val_sum = torch.sum(product_val,dim=-1)
        norm_val = torch.sum(weight,dim=-1)
        final_score = torch.div(product_val_sum,norm_val)
        final_score = final_score.view(B,-1)   # B x 1
        # final_score = torch.mean(score,dim=-1)
        # final_score = final_score.view(B, -1)
        return final_score

    def param_init(self,param_Enconder=None,param_MLP1=None,param_MLP2=None):
        if param_Enconder is not None:
            self.feature.load_state_dict(param_Enconder)
        if param_MLP1 is not None:
            self.score_mlp.load_state_dict(param_MLP1)
        if param_MLP2 is not None:
            self.weight_mlp.load_state_dict(param_MLP2)

class double_fusion(nn.Module):
    def __init__(self):
        super(double_fusion, self).__init__()
        self.score_compute=weight_score()

    def FusionLayer(self, x1, x2):
        difference = x1 - x2
        out = torch.div(1, 1 + torch.exp(-difference))
        return out
    def forward(self,x1,x2):
        '''
        :param x1:  B x patch_number x D x patch_size
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        '''
        score1 = self.score_compute(x1)
        score2 = self.score_compute(x2)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x = self.FusionLayer(score1,score2)     # B x 1
        return score1,score2,x
    def param_init(self,pth):
        self.score_compute.param_init(self,param_Enconder=pth)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    a=torch.randn(4,5,3,10).to(device)
    b=torch.randn(4,5,3,10).to(device)
    # print(a.device)
    # print(a)
    model=double_fusion().to(device)
    out = model(a,b)
    print("out[0].shape:",out[0].shape, out)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # torch.save(model.state_dict(),'params.pth')
    # pth = torch.load('params.pth')
    # print(pth['score_compute.feature.stn.conv1.weight'])
