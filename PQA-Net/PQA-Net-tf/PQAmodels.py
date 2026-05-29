import tensorflow as tf
import tensorflow_compression as tfc

class GDN(tf.keras.layers.Layer):
    def __init__(self,
        input_channel,
        inverse=False,
        data_format = "channel_last",
        # beta=None,
        # gamma=None,
        # beta_initializer=None,
        # gamma_initializer=None
        ):
        super(GDN, self).__init__()

        self.input_channel = input_channel
        self.inverse = inverse,
        self.data_format = data_format
        # self.beta = beta
        # self.gamma = gamma
        # self.beta_initializer = beta_initializer
        # self.gamma_initializer = gamma_initializer

    def build(self):
        c = self.input_channel
        self.beta = self.add_weight(name='beta',
                                    shape=(c),
                                    initializer="ones",
                                    trainable=True
                                    )

        self.gamma = self.add_weight(name="gamma",
                                    shape=(c, c),
                                    initializer="uniform",
                                    trainable=True
                                    )


class MeonDT(tf.keras.Model):
    def __init__(self, output_channel, clamp=False):
        super(MeonDT, self).__init__()

        self.output_channel = output_channel

        if clamp:
            pass
        else:
            self.clamp = None

        self.conv1 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=5, strides=2, padding="same",
        )
        self.gdn1 = tfc.GDN()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=5, strides=2, padding="same"
        )
        self.gdn2 = tfc.GDN()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, strides=2, padding="same"
        )
        self.gdn3 = tfc.GDN()

        self.conv4 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, padding="valid"
        )
        self.gdn4 = tfc.GDN()
        self.batchnorm = tf.keras.layers.BatchNormalization()

        self.st1_fc1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=1, strides=1, padding="valid"
        )
        self.st1_gdn = tfc.GDN()
        self.st1_fc2 = tf.keras.layers.Conv2D(
            filters=self.output_channel, kernel_size=1, strides=1, padding="valid"
        )

        # self.dropout = tf.keras.layers.Dropout(rate=0.5) 

    def call(self, inputs):
        batch_size = inputs.shape[0]

        feacon = []

        for _, proj_img in enumerate(tf.split(inputs, 6, 1)):
            proj_img = tf.cast(tf.squeeze(proj_img, axis=1), dtype=tf.float32)
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn1(self.conv1(proj_img)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn2(self.conv2(x)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn3(self.conv3(x)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn4(self.conv4(x)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")

            feacon.append(x)

        cc = tf.concat(feacon, axis=3)
        y = self.st1_gdn(self.st1_fc1(cc))
        y = self.st1_fc2(tf.nn.dropout(y, rate=0.5))
        y = tf.reshape(y, (batch_size, -1))

        if self.clamp:
            y = tf.clip_by_value(y, clip_value_min=0, clip_value_max=17)

        return y
    
 
class Meon(tf.keras.Model):
    def __init__(self, output_channel):
        super(Meon, self).__init__()

        self.output_channel = output_channel

        self.conv1 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=5, strides=2, padding="same"
        )
        self.gdn1 = tfc.GDN()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=5, strides=2, padding="same"
        )
        self.gdn2 = tfc.GDN()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, strides=2, padding="same"
        )
        self.gdn3 = tfc.GDN()

        self.conv4 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, padding="valid"
        )
        self.gdn4 = tfc.GDN()
        self.batchnorm = tf.keras.layers.BatchNormalization()

        self.st1_fc1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=1, strides=1, padding="valid"
        )
        self.st1_gdn = tfc.GDN()
        self.st1_fc2 = tf.keras.layers.Conv2D(
            filters=self.output_channel, kernel_size=1, strides=1, padding="valid"
        )

        # subtask 2
        self.st2_fc1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=1, strides=1, padding="valid"
        )
        self.st2_gdn = tfc.GDN()
        self.st2_fc2 = tf.keras.layers.Conv2D(
            filters=self.output_channel, kernel_size=1, strides=1, padding="valid"
        )


    def call(self, inputs):
        batch_size = inputs.shape[0]
        feacon = []

        for _, proj_img in enumerate(tf.split(inputs, 6, 1)):
            proj_img = tf.cast(tf.squeeze(proj_img, axis=1), dtype=tf.float32)
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn1(self.conv1(proj_img)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn2(self.conv2(x)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn3(self.conv3(x)), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")
            x = tf.nn.max_pool2d(tf.nn.dropout(self.gdn4(self.batchnorm(self.conv4(x))), rate=0.5),
                                 ksize=(2, 2), strides=2, padding="VALID")

            feacon.append(x)
        
        # task 1
        cc = tf.concat(feacon, axis=3)
        y = self.st1_gdn(self.st1_fc1(cc))
        y = self.st1_fc2(tf.nn.dropout(y, rate=0.5))
        y = tf.reshape(y, (batch_size, -1))
        
        # task 2
        p = tf.nn.softmax(y)
        p = tf.reshape(p, (batch_size, -1))

        s = self.st2_gdn(self.st2_fc1(tf.concat(feacon, axis=3)))
        s = self.st2_fc2(tf.nn.dropout(s, rate=0.5))
        s = tf.reshape(s, (batch_size, -1))

        g = tf.reduce_sum(p * s, axis=1)

        return y, g