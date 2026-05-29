import mindspore as ms
import mindspore.nn as nn
from .HSCNN import HSCNN 
import os

class Encoder(nn.Cell):
    def __init__(self, resume=None, backbone='HSCNN'):
        super(Encoder, self).__init__()
        if backbone == 'HSCNN':
            self.scnn = HSCNN()
        if resume != None and os.path.exists(resume):
            param_dict = ms.load_checkpoint(resume)
            param_not_load = ms.load_param_into_net(self.scnn, param_dict)
            print("scnn param_not_load: ", param_not_load)
        # if resume:
        #     self.scnn.load_state_dict(torch.load(resume),strict=False)
        self.__in_features = 256

    def construct(self, s_img1):
        dityFeat = self.scnn(s_img1)
        return dityFeat

    def output_num(self):
        return self.__in_features
