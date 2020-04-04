from tensorflow.keras.layers import (
    ZeroPadding2D, BatchNormalization, LeakyReLU, Conv2D, Add, Input, UpSampling2D, Concatenate, Lambda)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np


class V3Model:
    def __init__(self, input_shape, classes, anchors):
        self.current_layer = 1
        self.input_shape = input_shape
        self.classes = classes
        self.anchors = anchors 
        self.funcs = (ZeroPadding2D, BatchNormalization, LeakyReLU, Conv2D,
                      Add, Input, UpSampling2D, Concatenate, Lambda)
        self.func_names = [
            'zero_padding', 'batch_normalization', 'leaky_relu', 'conv2d',
            'add', 'input', 'up_sample', 'concat', 'lambda']
        self.layer_names = {func.__name__: f'layer_CURRENT_LAYER_{name}' for func, name in zip(
            self.funcs, self.func_names)}
        self.shortcuts = []

    def apply_func(self, func, x=None, *args, **kwargs):
        """
        Apply a function from self.funcs and increment layer count.
        Args:
            func: func from self.funcs.
            x: image tensor.
            *args: func args
            **kwargs: func kwargs

        Returns:
            result of func
        """
        name = self.layer_names[func.__name__].replace('CURRENT_LAYER', f'{self.current_layer}')
        result = func(name=name, *args, **kwargs)
        self.current_layer += 1
        if x is not None:
            return result(x)
        return result

    def convolution_block(self, x, filters, kernel_size, strides, batch_norm, action=None):
        """
        Convolution block for yolo version3.
        Args:
            x: Image input tensor.
            filters: Number of filters/kernels.
            kernel_size: Size of the filter/kernel.
            strides: The number of pixels a filter moves, like a sliding window.
            batch_norm: Standardizes the inputs to a layer for each mini-batch.
            action: 'add' or 'append'

        Returns:
            x or x added to shortcut.
        """
        if action == 'append':
            self.shortcuts.append(x)
        padding = 'same'
        if strides != 1:
            x = self.apply_func(ZeroPadding2D, x, padding=((1, 0), (1, 0)))
            padding = 'valid'
        x = self.apply_func(
            Conv2D, x, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=padding, use_bias=not batch_norm, kernel_regularizer=l2(0.0005))
        if batch_norm:
            x = self.apply_func(BatchNormalization, x)
            x = self.apply_func(LeakyReLU, x, alpha=0.1)
        if action == 'add':
            return self.apply_func(Add, [self.shortcuts.pop(), x])
        return x

    def make(self, training):
        input_initial = self.apply_func(Input, shape=self.input_shape)
        x = self.convolution_block(input_initial, 32, 3, 1, True)
        x = self.convolution_block(x, 64, 3, 2, True)
        x = self.convolution_block(x, 32, 1, 1, True, 'append')
        x = self.convolution_block(x, 64, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 3, 2, True)
        x = self.convolution_block(x, 64, 1, 1, True, 'append')
        x = self.convolution_block(x, 128, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 64, 1, 1, True, 'append')
        x = self.convolution_block(x, 128, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 3, 2, True)
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 128, 1, 1, True, 'append')
        x = self.convolution_block(x, 256, 3, 1, True, 'add')  #
        skip_36 = x
        x = self.convolution_block(x, 512, 3, 2, True)
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 256, 1, 1, True, 'append')
        x = self.convolution_block(x, 512, 3, 1, True, 'add')  #
        skip_61 = x
        x = self.convolution_block(x, 1024, 3, 2, True)
        x = self.convolution_block(x, 512, 1, 1, True, 'append')
        x = self.convolution_block(x, 1024, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 512, 1, 1, True, 'append')
        x = self.convolution_block(x, 1024, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 512, 1, 1, True, 'append')
        x = self.convolution_block(x, 1024, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 512, 1, 1, True, 'append')
        x = self.convolution_block(x, 1024, 3, 1, True, 'add')  #
        x = self.convolution_block(x, 512, 1, 1, True)
        x = self.convolution_block(x, 1024, 3, 1, True)
        x = self.convolution_block(x, 512, 1, 1, True)
        x = self.convolution_block(x, 1024, 3, 1, True)
        x = self.convolution_block(x, 512, 1, 1, True)  #
        x = self.convolution_block(x, 1024, 3, 1, True)
        x = self.convolution_block(x, 3 * (5 + self.classes), 1, 1, False)  #
        detection_0 = x
        output_0 = self.apply_func(Lambda, detection_0, lambda item: tf.reshape(item, (
            -1, tf.shape(item)[1], tf.shape(item)[2], 3, self.classes + 5)))
        x = self.convolution_block(x, 256, 1, 1, True)  #
        x = self.apply_func(UpSampling2D, x, size=2)
        x = self.apply_func(Concatenate, [x, skip_61])
        x = self.convolution_block(x, 256, 1, 1, True)
        x = self.convolution_block(x, 512, 3, 1, True)
        x = self.convolution_block(x, 256, 1, 1, True)
        x = self.convolution_block(x, 512, 3, 1, True)
        x = self.convolution_block(x, 256, 1, 1, True)  #
        x = self.convolution_block(x, 512, 3, 1, True)
        x = self.convolution_block(x, 3 * (5 + self.classes), 1, 1, False)  #
        detection_1 = x
        output_1 = self.apply_func(Lambda, detection_1, lambda item: tf.reshape(item, (
            -1, tf.shape(item)[1], tf.shape(item)[2], 3, self.classes + 5)))
        x = self.convolution_block(x, 128, 1, 1, True)  #
        x = self.apply_func(UpSampling2D, x, size=2)
        x = self.apply_func(Concatenate, [x, skip_36])
        x = self.convolution_block(x, 128, 1, 1, True)
        x = self.convolution_block(x, 256, 3, 1, True)
        x = self.convolution_block(x, 128, 1, 1, True)
        x = self.convolution_block(x, 256, 3, 1, True)
        x = self.convolution_block(x, 128, 1, 1, True)
        x = self.convolution_block(x, 256, 3, 1, True)
        x = self.convolution_block(x, 3 * (5 + self.classes), 1, 1, True)  #
        detection_2 = x
        output_2 = self.apply_func(Lambda, detection_2, lambda item: tf.reshape(item, (
            -1, tf.shape(item)[1], tf.shape(item)[2], 3, self.classes + 5)))
        if training:
            return Model(input_initial, [output_0, output_1, output_2])


if __name__ == '__main__':
    anc = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32)
    mod = V3Model((416, 416, 3), 80, anc).make(training=True)
    mod.summary()



