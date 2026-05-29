import tensorflow as tf
import tensorflow_addons as tfa

from models.Res_Blocks import BasicBlock

class ResSCNN(tf.keras.Model):
    CHANNELS = [3, 64, 64, 64, 64]
    # momentum of batchnorm in tensorflow is different from that in pytorch
    def __init__(self, bn_momentum=0.9):
        super(ResSCNN, self).__init__()
        CH = self.CHANNELS

        self.conv1 = tf.keras.layers.Conv3D(CH[1], strides=2, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.block1 = BasicBlock(CH[1], bn_momentum=bn_momentum)

        self.conv2 = tf.keras.layers.Conv3D(CH[2], strides=2, kernel_size=3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.block2 = BasicBlock(CH[2], bn_momentum=bn_momentum)

        self.conv3 = tf.keras.layers.Conv3D(CH[3], strides=2, kernel_size=3, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.block3 = BasicBlock(CH[3], bn_momentum=bn_momentum)

        self.conv4 = tf.keras.layers.Conv3D(CH[4], strides=2, kernel_size=3, padding="same")
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.block4 = BasicBlock(CH[4], bn_momentum=bn_momentum)

        self.glob_avg = tfa.layers.AdaptiveMaxPooling3D((1, 1, 1), data_format='channels_last')

        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, input):
        batch_size = input.shape[0]
        out_s1 = self.conv1(input)
        out_s1 = self.bn1(out_s1)
        out_s1 = tf.nn.relu(out_s1)
        out1 = self.block1(out_s1)

        out1_ = tf.reshape(self.glob_avg(out1), (batch_size, -1))

        out_s2 = self.conv2(out1)
        out_s2 = self.bn2(out_s2)
        out_s2 = tf.nn.relu(out_s2)
        out2 = self.block2(out_s2)

        out2_ = tf.reshape(self.glob_avg(out2), (batch_size, -1))

        out_s3 = self.conv3(out2)
        out_s3 = self.bn3(out_s3)
        out_s3 = tf.nn.relu(out_s3)
        out3 = self.block3(out_s3)

        out3_ = tf.reshape(self.glob_avg(out3), (batch_size, -1))

        out_s4 = self.conv4(out3)
        out_s4 = self.bn4(out_s4)
        out_s4 = tf.nn.relu(out_s4)
        out4 = self.block4(out_s4)

        out4_ = tf.reshape(self.glob_avg(out4), (batch_size, -1))

        out = tf.concat((out1_, out2_, out3_, out4_), 1)

        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

if __name__=="__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        
    model = ResSCNN()


    x = tf.random.uniform((1, 400, 400, 400, 3))

    model(x)