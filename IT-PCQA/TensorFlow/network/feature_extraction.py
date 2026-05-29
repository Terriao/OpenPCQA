import tensorflow as tf
from .HSCNN import HSCNN 

class Encoder(tf.keras.Model):
    def __init__(self, backbone='HSCNN'):
        super(Encoder, self).__init__()
        if backbone == 'HSCNN':
            self.scnn = HSCNN()
        self.__in_features = 256

    def call(self, s_img1, training_flag=True):
        dityFeat = self.scnn(s_img1, training_flag=training_flag)
        return dityFeat

    def output_num(self):
        return self.__in_features
