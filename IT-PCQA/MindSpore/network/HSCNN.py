import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal

# def weight_init(net): 
#     for m in net.modules():    
#         if isinstance(m, nn.Conv2d):         
#             nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
#             m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()

class HSCNN(nn.Cell):

    def __init__(self):
        super(HSCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64,64,3,2,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64,64,3,2,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64,64,3,2,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(64,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(64,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(64,64,3,2,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        # net = [self.conv1,self.bn1,self.relu1,self.conv2,self.bn2,self.relu2,self.conv3,self.bn3,self.relu3,
        #        self.conv4,self.bn4,self.relu4,self.conv5,self.bn5,self.relu5,self.conv6,self.bn6,self.relu6,
        #        self.conv7,self.bn7,self.relu7,self.conv8,self.bn8,self.relu8,self.conv9,self.bn9,self.relu9,]
        # for m in net:
        #     weight_init(m)
        self.pooling1 = nn.AvgPool2d(112,1)
        self.pooling2 = nn.AvgPool2d(56, 1)
        self.pooling3 = nn.AvgPool2d(28, 1)
        self.pooling4 = nn.AvgPool2d(14, 1)

        self.pooling_test = nn.AvgPool2d(218, 1)

        self.projection = nn.SequentialCell([
            nn.Conv2d(64*4,256,1,1,pad_mode='valid',has_bias=True, weight_init=HeNormal(nonlinearity='relu')), 
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,1,1,pad_mode='valid',has_bias=True, weight_init=HeNormal(nonlinearity='relu')), 
            nn.BatchNorm2d(256), nn.ReLU()])
        # weight_init(self.projection)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))

    def construct(self, X):
        N = X.shape[0]
        assert X.shape == (N, 3, 224, 224)

        X = self.conv1(X)
        X = self.bn1(X)
        X1 = self.relu1(X)

        X = self.conv2(X1)
        X = self.bn2(X)
        X2 = self.relu2(X)

        X = self.conv3(X2)
        X = self.bn3(X)
        X3 = self.relu3(X)

        X = self.conv4(X3)
        X = self.bn4(X)
        X4 = self.relu4(X)

        X = self.conv5(X4)
        X = self.bn5(X)
        X5 = self.relu5(X)

        X = self.conv6(X5)
        X = self.bn6(X)
        _,_,H,W = X4.shape
        X6 = self.relu6(X)

        X = self.conv7(X6)
        X = self.bn7(X)
        _, _, H, W = X3.shape
        X7 = self.relu7(X)

        X = self.conv8(X7)
        X = self.bn8(X)
        _, _, H, W = X2.shape
        X8 = self.relu8(X)

        X = self.conv9(X8)
        X = self.bn9(X)
        _, _, H, W = X1.shape
        X9 = self.relu9(X)

        out1 = self.pooling1(X3)
        out2 = self.pooling2(X5)
        out3 = self.pooling3(X7)
        out4 = self.pooling4(X9)

        X = ops.cat([out1, out2,out3,out4], 1)
        X = self.projection(X)
        X = X.view((X.shape[0], -1))

        return X
