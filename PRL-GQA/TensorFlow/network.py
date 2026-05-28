import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

class BasicFCModel(tf.keras.Model):
    def __init__(self):
        super(BasicFCModel, self).__init__()
        self.MLPLayers = tf.keras.Sequential([
            layers.Dense(512, kernel_initializer='he_uniform'),
            layers.BatchNormalization(axis=1),
            layers.ReLU(),
            layers.Dense(256, kernel_initializer='he_uniform'),
            layers.BatchNormalization(axis=1),
            layers.ReLU(),
            layers.Dense(64, kernel_initializer='he_uniform'),
            layers.BatchNormalization(axis=1),
            layers.ReLU(),
            layers.Dense(1, kernel_initializer='he_uniform')]
        )

    def call(self, inputs, training_flag=True):
        x = self.MLPLayers(inputs, training=training_flag)
        x = tf.keras.activations.sigmoid(x)
        return x


# class my_pointnet(tf.keras.Model):
#     def __init__(self):
#         super(my_pointnet, self).__init__()
#         self.conv1 = layers.Conv1D(filters=64, kernel_size=1)
#         self.conv2 = layers.Conv1D(filters=128, kernel_size=1)
#         self.conv3 = layers.Conv1D(filters=1024, kernel_size=1)
#         self.bn1 = layers.BatchNormalization()
#         self.bn2 = layers.BatchNormalization()
#         self.bn3 = layers.BatchNormalization()
#     def call(self,x):
#         x = tf.keras.activations.relu(self.bn1(self.conv1(x)))
#         x = tf.keras.activations.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = tf.math.reduce_max(x, 2, keepdim=True)  # B x 1024 X 1
#         x = tf.squeeze(x)  # B x 1024
#         return x

class MyPointnetDeep(tf.keras.Model):
    def __init__(self):
        super(MyPointnetDeep, self).__init__()
        self.conv1 = layers.Conv1D(filters=64, kernel_size=1, data_format='channels_first')
        self.conv2 = layers.Conv1D(filters=128, kernel_size=1, data_format='channels_first')
        self.conv3 = layers.Conv1D(filters=256, kernel_size=1, data_format='channels_first')
        self.conv4 = layers.Conv1D(filters=512, kernel_size=1, data_format='channels_first')
        self.bn1 = layers.BatchNormalization(axis=1)
        self.bn2 = layers.BatchNormalization(axis=1)
        self.bn3 = layers.BatchNormalization(axis=1)
        self.bn4 = layers.BatchNormalization(axis=1)

    def call(self,x, training_flag=True):
        '''

        :param x: B x D x number
        :return:
        '''
        x1 = tf.keras.activations.relu(self.bn1(self.conv1(x), training=training_flag))
        x2 = tf.keras.activations.relu(self.bn2(self.conv2(x1), training=training_flag))
        x3 = tf.keras.activations.relu(self.bn3(self.conv3(x2), training=training_flag))
        x4 = self.bn4(self.conv4(x3), training=training_flag)
        x1 = tf.math.reduce_max(x1, axis=2, keepdims=True)  # B x 64 X 1
        x1 = tf.squeeze(x1)
        x2 = tf.math.reduce_max(x2, axis=2, keepdims=True)  # B x 128 X 1
        x2 = tf.squeeze(x2)
        x3 = tf.math.reduce_max(x3, axis=2, keepdims=True)  # B x 256 X 1
        x3 = tf.squeeze(x3)
        x4 = tf.math.reduce_max(x4, axis=2, keepdims=True)  # B x 512 X 1
        x4 = tf.squeeze(x4)

        x = tf.concat([x1,x2],-1)
        x = tf.concat([x, x3],-1)
        x = tf.concat([x, x4],-1)   # B x 960
        return x

class WeightScore(tf.keras.Model):
    def __init__(self):
        super(WeightScore, self).__init__()

        self.feature = MyPointnetDeep()
        self.score_mlp = BasicFCModel()
        self.weight_mlp = BasicFCModel()
    def call(self,x, training_flag=True):
        '''
        :param x: B x patch_number x D x patch_size  // batch , patch, xyz, point_num
        :return: B x 1 //return batch*score
        '''
        B,patch_number,D,patch_size = x.shape

        x = tf.reshape(x,(-1,D,patch_size))    # pointnet网络输入只能接受3维    (B x patch_number) x D x patch_size
        fea_vector= self.feature(x,training_flag)    # (B x patch_number) x 1024
        score = self.score_mlp(fea_vector,training_flag)   # (B x patch_number) x 1
        weight = self.weight_mlp(fea_vector,training_flag)  # (B x patch_number) x 1  线性层只能接受2维
        score = tf.reshape(score, [B,patch_number])
        weight = tf.reshape(weight, [B,patch_number])   # B x patch_number
        product_val = tf.multiply(score,weight)
        product_val_sum = tf.reduce_sum(product_val,axis=-1)
        norm_val = tf.reduce_sum(weight,axis=-1)
        final_score = tf.divide(product_val_sum,norm_val)
        final_score = tf.reshape(final_score, [B,1])   # B x 1

        return final_score

    # def param_init(self,param_Enconder=None,param_MLP1=None,param_MLP2=None):
    #     if param_Enconder is not None:
    #         self.feature.load_weights(param_Enconder)
    #     if param_MLP1 is not None:
    #         self.score_mlp.load_weights(param_MLP1)
    #     if param_MLP2 is not None:
    #         self.weight_mlp.load_weights(param_MLP2)

class DoubleFusion(tf.keras.Model):
    def __init__(self):
        super(DoubleFusion, self).__init__()
        self.score_compute=WeightScore()

    def FusionLayer(self, x1, x2):
        difference = x1 - x2
        out = tf.divide(1, 1 + tf.exp(-difference))
        return out
    def call(self,x1,x2, training_flag=True):
        '''
        :param x1:  B x patch_number x D x patch_size
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        '''
        score1 = self.score_compute(x1,training_flag)
        score2 = self.score_compute(x2,training_flag)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x = self.FusionLayer(score1,score2)     # B x 1
        return score1,score2,x
    # def param_init(self,pth):
    #     self.score_compute.param_init(self,param_Enconder=pth)


if __name__ == '__main__':
    net = DoubleFusion()
    x1 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 64, 3, 516))
    x2 = tf.keras.initializers.RandomUniform(minval=0, maxval=5.0)(shape=(4, 64, 3, 516))
    dist1, dist2, out = net(x1,x2, training_flag=True)
