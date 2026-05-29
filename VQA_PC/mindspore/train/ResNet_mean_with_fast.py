
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal
import math

__all__ = ['ResNet', 'resnet50']

model_urls = {#先自行下载到本地
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50': 'https://download.mindspore.cn/models/r1.9/resnet50_ascend_v190_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, pad_mode="pad", stride=stride,
                     padding=dilation, group=groups, has_bias=False, dilation=dilation, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, pad_mode="pad", padding=0, stride=stride, has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'))


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.down_sample_layer layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.down_sample_layer = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample_layer is not None:
            identity = self.down_sample_layer(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.down_sample_layer layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample_layer = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample_layer is not None:
            identity = self.down_sample_layer(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, pad_mode="pad", stride=2, padding=3,
                               has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.SequentialCell([
              nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"), #如果 mode 为”CONSTANT”，使用0进行填充
              nn.MaxPool2d(kernel_size=3, stride=2)])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn_img = nn.BatchNorm1d(128)
        self.bn_video = nn.BatchNorm1d(128)
        self.avgpool = ops.ReduceMean(keep_dims=True) #nn.AdaptiveAvgPool2d((1, 1))
        self.adjust1 = nn.Dense(2048,128)
        self.adjust2 = nn.Dense(256,128)
        self.quality = nn.Dense(128+128,1)
        
        

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

        if zero_init_residual:
            for _, cell in self.cells_and_names():
                if isinstance(cell, Bottleneck) and cell.bn3.gamma is not None:
                    cell.bn3.gamma.set_data("zeros", cell.bn3.gamma.shape, cell.bn3.gamma.dtype)
                elif isinstance(cell, BasicBlock) and cell.bn2.gamma is not None:
                    cell.bn2.gamma.set_data("zeros", cell.bn2.gamma.shape, cell.bn2.gamma.dtype)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def quality_pred(self,in_channels,middle_channels,out_channels):
        regression_block = nn.SequentialCell([
            nn.Dense(in_channels, middle_channels),
            nn.Dense(middle_channels, out_channels),          
        ])

        return regression_block

    def hyper_structure1(self,in_channels,out_channels):

        hyper_block = nn.SequentialCell([
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1, pad_mode="pad",stride=1, padding=0,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3, pad_mode="pad",stride=1, padding=1,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1, pad_mode="pad",stride=1, padding=0,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
        ])

        return hyper_block

    def hyper_structure2(self,in_channels,out_channels):
        hyper_block = nn.SequentialCell([
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1, pad_mode="pad",stride=1, padding=0,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3, pad_mode="pad",stride=2, padding=1,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1, pad_mode="pad",stride=1, padding=0,has_bias=False, weight_init=HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu')),
        ])

        return hyper_block


    def construct(self, x, x_fast_features):
        # See note [TorchScript super()]
        x_size = x.shape
        x_fast_features_size = x_fast_features.shape
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x_fast_features = x_fast_features.view(-1, x_fast_features_size[2])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)

        x_avg = self.avgpool(x,[i for i in range(len(x.shape))][-2:])
        # x = torch.flatten(x_avg, 1)
        x = ops.Flatten()(x_avg)
        # x = torch.cat((self.bn_img((self.adjust1(x))), self.bn_video(self.adjust2(x_fast_features))), dim=1)
        x = ops.cat((self.bn_img((self.adjust1(x))), self.bn_video(self.adjust2(x_fast_features))), axis=1)

        output = self.quality(x)
        output = output.view(x_size[0],x_size[1])
        # output = torch.mean(output,dim=1)
        output = ops.ReduceMean(keep_dims=False)(output, 1)
        return output

def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

def resnet50(pretrained=False, ckpt_path='/userhome/VQA_PC-main/resnet50_ascend_v190_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt', **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        ckpt_path (str): The path of pretrained model
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # ms_param = mindspore_params(model)
    if pretrained:
        param_dict = ms.load_checkpoint(ckpt_path)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print("param_not_load: ", param_not_load) #打印网络中没有被加载的参数
        print ('load the pretrained model, done!')
    return model

if __name__ == '__main__':
    net = resnet50(pretrained=True)
    video = ops.StandardNormal(seed=2)((1,4,3,448,448))
    features = ops.StandardNormal(seed=2)((1,4,256))
    print(net(video,features))
    