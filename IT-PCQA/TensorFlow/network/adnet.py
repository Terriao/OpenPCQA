import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class AdversarialNetwork(tf.keras.Model):
  def __init__(self):
    super(AdversarialNetwork, self).__init__()

    self.ad_layer3 = tf.keras.Sequential([layers.Dense(64, kernel_initializer='he_uniform'),
                                   layers.Dense(1, kernel_initializer='he_uniform'),
                                   ])
    self.sigmoid = tf.keras.activations.sigmoid

  def call(self, xfeature, yout, D_s, D_t, source_size, target_size): #D_t=0
    y = self.ad_layer3(xfeature)
    y = self.sigmoid(y)
    dc_target = tf.constant(np.array([[1]] * source_size + [[0]] * target_size), dtype=tf.float32)
    Dfake = tf.constant(np.array([[D_s]] * source_size + [[D_t]] * target_size), dtype=tf.float32)
    y = tf.math.abs(y - Dfake)
    return tf.keras.losses.BinaryCrossentropy(from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(dc_target, y) #对结果求mean
