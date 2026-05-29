import tensorflow as tf
import tensorflow.keras.layers as layers

class Feature_mapping(tf.keras.Model):
  def __init__(self, hidden_size):
    super(Feature_mapping, self).__init__()
    self.ad_layer1 = layers.Dense(hidden_size, kernel_initializer='he_uniform')
    self.ad_layer2 = layers.Dense(hidden_size, kernel_initializer='he_uniform')
    self.relu1 = tf.nn.relu
    self.relu2 = tf.nn.relu

  def call(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    return x
