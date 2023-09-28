from keras import backend as K
import tensorflow

from keras.losses import categorical_crossentropy, binary_crossentropy
import tensorflow as tf
lambda_rpn_class=1
def rpn_loss_cls_new(y_true, y_pred):
    """
    Categorical cross-entropy loss, adaptated to normalize by the number of valid samples (samples assigned to a class, not ignored)
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    # print(K.sum(binary_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
    #     K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2)))

    return lambda_rpn_class * K.sum(categorical_crossentropy(y_true, y_pred), axis=(0,1,2)) / K.sum(
        K.cast(K.greater(categorical_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0,1, 2))




lambda_rpn_reg=0.01
def rpn_loss_reg(y_true, y_pred):
    """
    Leaky L1 norm loss, adaptated to exclude ignored pixels
    :param y_true: ground truth output [ndarray]
    :param y_pred: predicted output [ndarray]
    :return: loss function value []
    """

    mask = K.cast(K.not_equal(y_true, 0), 'float32')

    aaa = y_true - y_pred
    aaa = aaa*mask

    x_abs = K.abs(aaa)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

    # return lambda_rpn_reg * K.sum(x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5), axis=(1, 2, 3))
    return lambda_rpn_reg * K.sum( (x_bool * (0.5 * aaa * aaa) + (1 - x_bool) * (x_abs - 0.5)), axis=(0, 1, 2, 3))

