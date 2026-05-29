import mindspore.nn as nn
from network.feature_extraction import Encoder
from network.feature_mapping import Feature_mapping
from network.regression import Regression
from network.adnet import AdversarialNetwork
from scipy.stats import spearmanr
import mindspore.ops as ops

class ITModel(nn.Cell):
    def __init__(self, channel,backbone='HSCNN'):
        super(ITModel, self).__init__()
        self.extraction = Encoder(backbone = backbone)
        self.mapping = Feature_mapping(channel, channel)
        self.regression = Regression(channel)
        self.adnet = AdversarialNetwork(channel)

    def construct(self, s_img1,data_source=None,label_source=None,data_target=None,is_train=False): #model = [extraction, mapping, regression, adnet] source是图片，target是点云
        feature1 = self.extraction(s_img1) #feature1.shape: (N, 256)
        feature2 = self.mapping(feature1) #feature2.shape: (N, 256)
        score = self.regression(feature2) #score.shape: (N, 1)

        if is_train:
            score1 = self.regression(feature1)
            srocc_latter, _ = spearmanr((score.narrow(0, 0, data_source.shape[0])[:, 0]).asnumpy(), 
                                        label_source.asnumpy())
            srocc_former, _ = spearmanr((score1.narrow(0, 0, data_source.shape[0])[:, 0]).asnumpy(),
                                        label_source.asnumpy())
            if srocc_latter > srocc_former + 0.1:
                Dfake_source = 1
            else:
                Dfake_source = 0

            Dfake_target = 0

            loss3 = self.adnet(feature2, score, Dfake_source, Dfake_target, data_source.shape[0], data_target.shape[0])
            return loss3, score
        else:
            return score

if __name__ == '__main__':
    backbone = 'HSCNN'
    pic_resize = 224
    channel = 256
    model = ITModel(channel,backbone=backbone)
    img = ops.UniformReal(seed=2)((2, 3, 224, 224))
    # print(img)
    feature1 = model.extraction(img)
    feature2 = model.mapping(feature1)
    score = model.regression(feature2)
    print("feature1.shape:",feature1.shape)
    print("feature2.shape:",feature2.shape)
    print("score.shape:",score.shape)
    