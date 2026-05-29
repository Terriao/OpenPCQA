import tensorflow as tf
import tensorflow.keras.layers as layers

class Regression(tf.keras.Model):
    def __init__(self):
        super(Regression, self).__init__()
        self.regression = tf.keras.Sequential([layers.Dense(128, kernel_initializer='he_uniform'),
                                        layers.Dense(1, kernel_initializer='he_uniform')])

    def call(self, s_img1):
        out = self.regression(s_img1)
        return out
