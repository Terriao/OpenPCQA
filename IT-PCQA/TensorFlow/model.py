import tensorflow as tf
import tensorflow.keras.layers as layers
from network.feature_extraction import Encoder
from network.feature_mapping import Feature_mapping
from network.regression import Regression
from network.adnet import AdversarialNetwork
from scipy.stats import spearmanr

class ITModel(tf.keras.Model):
    def __init__(self, channel,backbone='HSCNN'):
        super(ITModel, self).__init__()
        self.extraction = Encoder(backbone = backbone)
        self.mapping = Feature_mapping(channel)
        self.regression = Regression()
        self.adnet = AdversarialNetwork()

    def call(self, s_img1,data_source=None,label_source=None,data_target=None,is_train=False): #model = [extraction, mapping, regression, adnet] source是图片，target是点云
        feature1 = self.extraction(s_img1, training_flag=is_train) #feature1.shape: (N, 256)
        feature2 = self.mapping(feature1) #feature2.shape: (N, 256)
        score = self.regression(feature2) #score.shape: (N, 1)

        if is_train:
            score1 = self.regression(feature1)
            srocc_latter, _ = spearmanr(score[:data_source.shape[0], 0].numpy(), 
                                        label_source.numpy())
            srocc_former, _ = spearmanr(score1[:data_source.shape[0], 0].numpy(),
                                        label_source.numpy())
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
    model = ITModel(channel,backbone=backbone) #到这里
    model.build(input_shape = (2, 3, 224, 224))
    img = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(8, 3, 224, 224))
    data_source = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    data_target = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 3, 224, 224))
    label_source = tf.keras.initializers.RandomUniform(minval=0, maxval=1.0)(shape=(4,))
    print(img)
    feature1 = model.extraction(img)
    feature2 = model.mapping(feature1)
    score = model.regression(feature2)
    print("feature1.shape:",feature1.shape)
    print("feature2.shape:",feature2.shape)
    print("score.shape:",score.shape)
    loss3, score = model(img,data_source,label_source,data_target,is_train=True)
    print("loss3:",loss3)
    print("score.shape:",score.shape)
