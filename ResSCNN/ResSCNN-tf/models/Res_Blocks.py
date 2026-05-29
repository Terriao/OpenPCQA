import tensorflow as tf


class BasicBlock(tf.keras.Model):
    def __init__(self, planes, bn_momentum=0.9, down_sample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(planes, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.conv2 = tf.keras.layers.Conv3D(planes, kernel_size=3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.downsample = down_sample

    def call(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = tf.nn.relu(out)

        return out



if __name__=="__main__":
    x = tf.random.uniform((1, 400, 400, 400, 3))
    block = BasicBlock(3)
    block(x)
