import tensorflow as tf

from tensorflow.keras import layers, Model

class Modelin(Model):
    def __init__(self):
        super(Modelin, self).__init__()
        self.prep_one = layers.RandomFlip()
        self.prep_two = layers.RandomTranslation(0, 0.2)

    def call(self, data):
        inputs = self.prep_one(data)
        output = self.prep_two(inputs)
        return output

class Modeler(Model):
    def __init__(self, output_node):
        super(Modeler, self).__init__()
        self.conv_one = layers.Conv2D(16, 3, activation=tf.nn.relu, padding="same")
        self.pool_one = layers.MaxPool2D((2, 2))
        self.conv_two = layers.Conv2D(32, 3, activation=tf.nn.relu, padding="same")
        self.pool_two = layers.MaxPool2D((2, 2))
        self.conv_thr = layers.Conv2D(64, 3, activation=tf.nn.relu, padding="same")
        self.pool_thr = layers.MaxPool2D((2, 2))

        self.flat_hid = layers.Flatten()
        self.drop_hid = layers.Dropout(0.2)
        self.dens_one = layers.Dense(256, activation=tf.nn.relu)
        self.last_out = layers.Dense(output_node, activation=tf.nn.softmax)

    def call(self, data):
        inputs = self.conv_one(data)
        x = self.pool_one(inputs)
        x = self.conv_two(x)
        x = self.pool_two(x)
        x = self.conv_thr(x)
        x = self.pool_thr(x)

        x = self.flat_hid(x)
        x = self.drop_hid(x)
        x = self.dens_one(x)
        output = self.last_out(x)
        return output