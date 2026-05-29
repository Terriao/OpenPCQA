import mindspore.nn as nn
import mindspore as ms
from .Res_blocks import BasicBlock
    
class ResSCNN(nn.Cell):
    # momentum of batchnorm in tensorflow is different from that in pytorch
    def __init__(self, bn_momentum=0.9):
        super(ResSCNN, self).__init__()
        CH = [3, 64, 64, 64, 64]

        self.conv1 = ms.nn.Conv3d(3, CH[1], stride=2, kernel_size=3, pad_mode="same")
        self.bn1 = ms.nn.BatchNorm3d(CH[1], momentum=bn_momentum)

        self.block1 = BasicBlock(CH[1], bn_momentum=bn_momentum)

        self.conv2 = ms.nn.Conv3d(CH[1], CH[2], stride=2, kernel_size=3, pad_mode="same")
        self.bn2 = ms.nn.BatchNorm3d(CH[2], momentum=bn_momentum)

        self.block2 = BasicBlock(CH[2], bn_momentum=bn_momentum)

        self.conv3 = ms.nn.Conv3d(CH[2], CH[3], stride=2, kernel_size=3, pad_mode="same")
        self.bn3 = ms.nn.BatchNorm3d(CH[3], momentum=bn_momentum)

        self.block3 = BasicBlock(CH[3], bn_momentum=bn_momentum)

        self.conv4 = ms.nn.Conv3d(CH[3], CH[4], stride=2, kernel_size=3, pad_mode="same")
        self.bn4 = ms.nn.BatchNorm3d(CH[4], momentum=bn_momentum)

        self.block4 = BasicBlock(CH[4], bn_momentum=bn_momentum)

        self.glob_avg = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Dense(64+64+64+64, 32)
        self.fc2 = nn.Dense(32, 1)

    def construct(self, input):
        batch_size = input.shape[0]
        out_s1 = self.conv1(input)
        out_s1 = self.bn1(out_s1)
        out_s1 =ms.ops.relu(out_s1)
        out1 = self.block1(out_s1)

        out1_ = ms.ops.reshape(self.glob_avg(out1), (batch_size, -1))

        out_s2 = self.conv2(out1)
        out_s2 = self.bn2(out_s2)
        out_s2 = ms.ops.relu(out_s2)
        out2 = self.block2(out_s2)

        out2_ = ms.ops.reshape(self.glob_avg(out2), (batch_size, -1))

        out_s3 = self.conv3(out2)
        out_s3 = self.bn3(out_s3)
        out_s3 = ms.ops.relu(out_s3)
        out3 = self.block3(out_s3)

        out3_ = ms.ops.reshape(self.glob_avg(out3), (batch_size, -1))

        out_s4 = self.conv4(out3)
        out_s4 = self.bn4(out_s4)
        out_s4 = ms.ops.relu(out_s4)
        out4 = self.block4(out_s4)

        out4_ = ms.ops.reshape(self.glob_avg(out4), (batch_size, -1))

        out = ms.ops.concat((out1_, out2_, out3_, out4_), 1)

        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

    
    
if __name__=="__main__":
    import numpy as np
    model = ResSCNN()

    x = ms.Tensor(np.ones((1, 3, 400, 400, 400)), dtype=ms.float32)

    model(x)