
import math
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.layers as layers

__all__ = ['ResNet', 'resnet50']

x = layers.Input(shape=(224,224,3), batch_size=None, dtype=tf.float32)
# conv_base = ResNet50(weights='imagenet', include_top=False, input_tensor=x, input_shape=(224, 224, 3),pooling='avg') #如果不加载预训练参数，weights=None
class ResNet(tf.keras.Model):
    def __init__(self, pretrained=False, ckpt_path='/userhome/VQA_PC-main/tensorflow/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        super().__init__()
        conv_base = ResNet50(weights=None, include_top=False, input_tensor=None, input_shape=(224, 224, 3),pooling='avg') #如果不加载预训练参数，weights=None
        conv_base.trainable = True
        if pretrained:
            conv_base.load_weights(ckpt_path)#前面不加载参数，这里通过这种方式加载，实测有效，加载前后参数确实变了
        self.model = tf.keras.models.Sequential()
        self.model.add(conv_base)
        self.model.add(layers.Dense(128, kernel_initializer='he_uniform')) #对应adjust1
        self.model.add(layers.BatchNormalization(axis=-1, center=True, scale=True)) #对应bn_img
        self.model.build([None] + [224,224] + [3])
        self.adjust2 = layers.Dense(128, kernel_initializer='he_uniform')
        self.bn_video = layers.BatchNormalization(axis=-1, center=True, scale=True)
        self.quality = layers.Dense(1, kernel_initializer='he_uniform')

    def call(self, x, x_fast_features, training_flag=True):
        x_size = x.shape
        x_fast_features_size = x_fast_features.shape
        x = tf.reshape(x,[-1, x_size[2], x_size[3], x_size[4]])
        x_fast_features = tf.reshape(x_fast_features, [-1, x_fast_features_size[2]])
        x = self.model(x, training=training_flag)
        x_fast_features = self.bn_video(self.adjust2(x_fast_features), training=training_flag)
        x = tf.concat([x, x_fast_features], axis=1)
        output = self.quality(x)
        output = tf.reshape(output,[x_size[0],x_size[1]])
        output = tf.math.reduce_mean(output, axis=1, keepdims=False)
        return output

# checkpoint = tf.train.Checkpoint(model_net=model)
# checkpoint.save('/userhome/VQA_PC-main/tensorflow/' + 'model.ckpt')
# latest_ckpt = tf.train.latest_checkpoint('/userhome/VQA_PC-main/tensorflow/')
# checkpoint.restore(latest_ckpt)

if __name__ == '__main__':
    # net = ResNet(pretrained=True)
    input_data = tf.random.uniform(shape=[8,4, 224, 224, 3],minval=0,maxval=1.0,dtype=tf.float32)
    features = tf.random.uniform(shape=[8,4,256],minval=0,maxval=2.0,dtype=tf.float32)
    # print(net(input_data,features))
    ckpt = '/userhome/VQA_PC-main/tensorflow/train/ResNet_mean_with_fast_SJTU_5_best.h5'
    # net.save_weights(ckpt)
    net1 = ResNet(pretrained=False)
    print(net1(input_data,features))#要先call一遍，才能load_weights
    net1.load_weights(ckpt)