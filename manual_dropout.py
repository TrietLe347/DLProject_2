import tensorflow as tf


class ManualDropout(tf.keras.layers.Layer):

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def call(self, inputs, training = False):

        if training:
            mask = tf.random.uniform(tf.shape(inputs)) >= self.rate
            return tf.cast(mask,inputs.dtype) * inputs / (1.0 - self.rate)

        return inputs