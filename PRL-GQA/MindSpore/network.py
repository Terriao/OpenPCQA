import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
import os
# from pointnet_utils import PointNetEncoder
# import torch.nn.functional as F

esp = 1e-8
class BasicFCModule(nn.Cell):
    def __init__(self, inp_len=1024):
        super(BasicFCModule, self).__init__()
        self.MLPLayers1 = nn.SequentialCell([
            nn.Dense(inp_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dense(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dense(64, 1),
            nn.Sigmoid(),
        ])

    def construct(self, x):
        '''
        :param x:   N x C
        :return:    N x 1
        '''
        x = self.MLPLayers1(x)
        return x

# class my_pointnet(nn.Module):
#     def __init__(self,channel=3):
#         super(my_pointnet, self).__init__()
#         self.conv1 = nn.Conv1d(channel, 64, 1, has_bias=True, pad_mode='valid')
#         self.conv2 = nn.Conv1d(64, 128, 1, has_bias=True, pad_mode='valid')
#         self.conv3 = nn.Conv1d(128, 1024, 1, has_bias=True, pad_mode='valid')
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

class MyPointnetDeep(nn.Cell):
    def __init__(self,channel=3):
        super(MyPointnetDeep, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1, has_bias=True, pad_mode='valid')
        self.conv2 = nn.Conv1d(64, 128, 1, has_bias=True, pad_mode='valid')
        self.conv3 = nn.Conv1d(128, 256, 1, has_bias=True, pad_mode='valid')
        self.conv4 = nn.Conv1d(256, 512, 1, has_bias=True, pad_mode='valid')
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
    def construct(self,x):
        '''

        :param x: B x D x number
        :return:
        '''
        x = self.conv1(x)
        B, D, N = x.shape
        x = x.transpose((0,2,1)).reshape((-1, D))
        x1 = self.relu(self.bn1(x))
        x1 = x1.reshape((B, N, D)).transpose((0,2,1))

        x1t = self.conv2(x1)
        B, D, N = x1t.shape
        x1t = x1t.transpose((0,2,1)).reshape((-1, D))
        x2 = self.relu(self.bn2(x1t))
        x2 = x2.reshape((B, N, D)).transpose((0,2,1))

        x2t = self.conv3(x2)
        B, D, N = x2t.shape
        x2t = x2t.transpose((0,2,1)).reshape((-1, D))
        x3 = self.relu(self.bn3(x2t))
        x3 = x3.reshape((B, N, D)).transpose((0,2,1))

        x3t = self.conv4(x3)
        B, D, N = x3t.shape
        x3t = x3t.transpose((0,2,1)).reshape((-1, D))
        x4 = self.bn4(x3t)
        x4 = x4.reshape((B, N, D)).transpose((0,2,1))

        x1 = ops.ArgMaxWithValue(axis=2, keep_dims=True)(x1)[1]
        # x1 = torch.max(x1, 2, keepdim=True)[0]  # B x 64 X 1
        x1 = x1.view(-1, 64)
        x2 = ops.ArgMaxWithValue(axis=2, keep_dims=True)(x2)[1]
        # x2 = torch.max(x2, 2, keepdim=True)[0]  # B x 128 X 1
        x2 = x2.view(-1, 128)
        x3 = ops.ArgMaxWithValue(axis=2, keep_dims=True)(x3)[1]
        # x3 = torch.max(x3, 2, keepdim=True)[0]  # B x 256 X 1
        x3 = x3.view(-1, 256)
        x4 = ops.ArgMaxWithValue(axis=2, keep_dims=True)(x4)[1]
        # x4 = torch.max(x4, 2, keepdim=True)[0]  # B x 512 X 1
        x4 = x4.view(-1, 512)

        x = ops.concat((x1,x2),-1)
        x = ops.concat((x, x3), -1)
        x = ops.concat((x, x4), -1)   # B x 960


        #x = ops.concat((x3, x4), -1)   # B x 960
        return x

class WeightScore(nn.Cell):
    def __init__(self):
        super(WeightScore, self).__init__()
        # self.feature=PointNetEncoder(global_feat=True,feature_transform=True)
        # self.feature =my_pointnet()
        # self.score_mlp=BasicFCModule(1024)
        # self.weight_mlp=BasicFCModule(1024)
        self.feature = MyPointnetDeep()
        self.score_mlp = BasicFCModule(960)
        self.weight_mlp = BasicFCModule(960)
    def construct(self,x):
        '''
        :param x: B x patch_number x D x patch_size  // batch , patch, xyz, point_num
        :return: B x 1 //return batch*score
        '''
        B,patch_number,D,patch_size = x.shape
        x = x.view(-1,D,patch_size)    # pointnet网络输入只能接受3维    (B x patch_number) x D x patch_size
        fea_vector= self.feature(x)    # (B x patch_number) x 960
        score = self.score_mlp(fea_vector)   # (B x patch_number) x 1
        weight = self.weight_mlp(fea_vector)  # (B x patch_number) x 1  线性层只能接受2维
        score = score.view(B,patch_number)
        weight = weight.view(B,patch_number)   # B x patch_number
        product_val = ops.mul(score,weight)
        product_val_sum = ops.ReduceSum()(product_val,axis=-1)
        norm_val = ops.ReduceSum()(weight,axis=-1)
        final_score = ops.div(product_val_sum,norm_val)
        final_score = final_score.view(B,-1)   # B x 1
        # final_score = torch.mean(score,dim=-1)
        # final_score = final_score.view(B, -1)
        return final_score

    # def param_init(self,param_Enconder=None,param_MLP1=None,param_MLP2=None):
    #     if param_Enconder is not None:
    #         self.feature.load_state_dict(param_Enconder)
    #     if param_MLP1 is not None:
    #         self.score_mlp.load_state_dict(param_MLP1)
    #     if param_MLP2 is not None:
    #         self.weight_mlp.load_state_dict(param_MLP2)

class DoubleFusion(nn.Cell):
    def __init__(self):
        super(DoubleFusion, self).__init__()
        self.score_compute = WeightScore()

    def FusionLayer(self, x1, x2):
        difference = x1 - x2
        out = ops.div(1, 1 + ops.exp(-difference))
        return out

    def construct(self,x1,x2):
        '''
        :param x1:  B x patch_number x D x patch_size (B, S=64个中心点, 3, patch_size=516=每个中心点的邻近点数)
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        '''
        score1 = self.score_compute(x1)
        score2 = self.score_compute(x2)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x = self.FusionLayer(score1,score2)     # B x 1
        return score1,score2,x
    
    # def param_init(self,pth):
    #     self.score_compute.param_init(self,param_Enconder=pth)

if __name__ == '__main__':
    ms.set_context(device_target="GPU")
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE(静态图模式) PYNATIVE_MODE(动态图模式)
    context.set_context(save_graphs=False)
    context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
    if ms.get_context("device_target") == "GPU":
        context.set_context(enable_graph_kernel=False) #开启图算融合以优化网络执行性能，常用于GPU，动静态模式应该都可以
    if ms.get_context("mode") == ms.PYNATIVE_MODE:
        ms.set_context(mempool_block_size="1GB") #设置设备内存池的块大小，实际使用的内存池块大小是设备的可用内存和 mempool_block_size 值中的最小值
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)

    a=ms.numpy.randn((4,5,3,10)) #batch , patch, xyz, point_num
    b=ms.numpy.randn((4,5,3,10)) #batch , patch, xyz, point_num
    print(a.shape)
    model=DoubleFusion().set_grad()
    out = model(a,b)
    print("out[0].shape:",out[0].shape, out)
    for param in model.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)

    '''
    import torch
    import numpy  as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = np.random.randn(4,6,8).astype(np.float32)
    xt = torch.tensor(a).to(device)
    ot = torch.nn.BatchNorm1d(6).to(device)(xt)
    ot = torch.nn.functional.relu(ot)
    print("ot:",ot.shape,ot)

    xm = ms.Tensor(a)
    B, D, N = xm.shape
    xm = xm.transpose((0,2,1)).reshape((-1, D))
    print(xm.shape)
    om = ms.nn.BatchNorm1d(D).set_train()(xm)
    om = nn.ReLU()(om)
    om = om.reshape((B, N, D)).transpose((0,2,1))
    print("om:",om.shape,om)

    import tensorflow as tf
    xf = tf.convert_to_tensor(a)
    of = tf.keras.layers.BatchNormalization(axis=1)(xf, training=True)
    of = tf.keras.layers.ReLU()(of)
    print("of:",of.shape,of)
    '''