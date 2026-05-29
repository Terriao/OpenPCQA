import tensorflow as tf

class HSCNN(tf.keras.Model):

    def __init__(self):
        super(HSCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64,3,strides=(1, 1),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        # self.conv1 = nn.Conv2d(3,64,3,1,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1) #先用TF的默认momentum看看 , momentum=0.9, epsilon=1e-5
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = tf.nn.relu

        self.conv2 = tf.keras.layers.Conv2D(64,3,strides=(2, 2),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        # self.conv2 = nn.Conv2d(64,64,3,2,pad_mode="pad",padding=1,has_bias=True, weight_init=HeNormal(nonlinearity='relu'))
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu2 = tf.nn.relu

        self.conv3 = tf.keras.layers.Conv2D(64,3,strides=(1, 1),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn3 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu3 = tf.nn.relu

        self.conv4 = tf.keras.layers.Conv2D(64,3,strides=(2, 2),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn4 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu4 = tf.nn.relu

        self.conv5 = tf.keras.layers.Conv2D(64,3,strides=(1, 1),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn5 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu5 = tf.nn.relu

        self.conv6 = tf.keras.layers.Conv2D(64,3,strides=(2, 2),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn6 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu6 = tf.nn.relu

        self.conv7 = tf.keras.layers.Conv2D(64,3,strides=(1, 1),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn7 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu7 = tf.nn.relu

        self.conv8 = tf.keras.layers.Conv2D(64,3,strides=(1, 1),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn8 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu8 = tf.nn.relu

        self.conv9 = tf.keras.layers.Conv2D(64,3,strides=(2, 2),padding="same",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn9 = tf.keras.layers.BatchNormalization(axis=1)
        self.relu9 = tf.nn.relu

        self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=(112, 112), strides=1, padding='valid', data_format='channels_first')
        # self.pooling1 = nn.AvgPool2d(112,1)
        self.pooling2 = tf.keras.layers.AveragePooling2D(pool_size=(56, 56), strides=1, padding='valid', data_format='channels_first')
        self.pooling3 = tf.keras.layers.AveragePooling2D(pool_size=(28, 28), strides=1, padding='valid', data_format='channels_first')
        self.pooling4 = tf.keras.layers.AveragePooling2D(pool_size=(14, 14), strides=1, padding='valid', data_format='channels_first')

        self.pooling_test = tf.keras.layers.AveragePooling2D(pool_size=(218, 218), strides=1, padding='valid', data_format='channels_first')

        self.projection = [
            tf.keras.layers.Conv2D(256,1,strides=(1, 1),padding="valid",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal()),
            # nn.Conv2d(64*4,256,1,1,pad_mode='valid',has_bias=True, weight_init=HeNormal(nonlinearity='relu')), 
            tf.keras.layers.BatchNormalization(axis=1), tf.nn.relu,
            tf.keras.layers.Conv2D(256,1,strides=(1, 1),padding="valid",
            data_format='channels_first',use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal()), 
            tf.keras.layers.BatchNormalization(axis=1), tf.nn.relu]
        # weight_init(self.projection)
        # for _, cell in self.cells_and_names():
        #     if isinstance(cell, nn.BatchNorm2d):
        #         cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
        #         cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))

    def call(self, X, training_flag=True):
        N = X.shape[0]
        assert X.shape == (N, 3, 224, 224)

        X = self.conv1(X)
        X = self.bn1(X, training=training_flag)
        X1 = self.relu1(X)

        X = self.conv2(X1)
        X = self.bn2(X, training=training_flag)
        X2 = self.relu2(X)

        X = self.conv3(X2)
        X = self.bn3(X, training=training_flag)
        X3 = self.relu3(X)

        X = self.conv4(X3)
        X = self.bn4(X, training=training_flag)
        X4 = self.relu4(X)

        X = self.conv5(X4)
        X = self.bn5(X, training=training_flag)
        X5 = self.relu5(X)

        X = self.conv6(X5)
        X = self.bn6(X, training=training_flag)
        _,_,H,W = X4.shape
        X6 = self.relu6(X)

        X = self.conv7(X6)
        X = self.bn7(X, training=training_flag)
        _, _, H, W = X3.shape
        X7 = self.relu7(X)

        X = self.conv8(X7)
        X = self.bn8(X, training=training_flag)
        _, _, H, W = X2.shape
        X8 = self.relu8(X)

        X = self.conv9(X8)
        X = self.bn9(X, training=training_flag)
        _, _, H, W = X1.shape
        X9 = self.relu9(X)

        out1 = self.pooling1(X3)
        out2 = self.pooling2(X5)
        out3 = self.pooling3(X7)
        out4 = self.pooling4(X9)

        X = tf.concat([out1, out2,out3,out4], axis=1)
        for module in self.projection:
            if isinstance(module, tf.keras.layers.BatchNormalization):
                X = module(X, training=training_flag)
            else:
                X = module(X)
        # X = self.projection(X)
        X = tf.reshape(X,[X.shape[0], -1])

        return X
