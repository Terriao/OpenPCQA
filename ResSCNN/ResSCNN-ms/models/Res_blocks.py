import mindspore as ms
import mindspore.nn as nn

class BasicBlock(nn.Cell):
    def __init__(self, planes, bn_momentum=0.9, down_sample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ms.nn.Conv3d(planes, planes, kernel_size=3, pad_mode="same")
        self.bn1 = ms.nn.BatchNorm3d(planes, momentum=bn_momentum)

        self.conv2 = ms.nn.Conv3d(planes, planes, kernel_size=3, pad_mode="same")
        self.bn2 = ms.nn.BatchNorm3d(planes, momentum=bn_momentum)

        self.downsample = down_sample

    def construct(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        out = ms.ops.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = ms.ops.relu(out)

        return out

