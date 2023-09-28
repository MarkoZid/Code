from keras.layers import Conv2D, MaxPool2D, Input, concatenate,Cropping2D
from keras.models import Model, model_from_json

def load_model(model_path, weights_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :param weights_path: path of the serialized model weights file (.h5) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    model.load_weights(weights_path)     # load weights

    return model


def construct_model_anchorless_detector_Pedestrians_anchorless(input_shape):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """
    #NOTE 56px receptive field so cekor 4(bez tretiot maxpooling

    #NOTE 84px receptive field so cekor 8

    input_layer = Input(shape=input_shape)


    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv1')(
        input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv2')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv4')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv5')(f4)

    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv7')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv8')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    # f9=f8


    # f10 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_1 = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    # f10_con=concatenate([f10_1, f10], axis=3)



    # f11 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f10)

    f10 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv10')(f9)
    f11= Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv11')(f10)

    x_class_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv12')(f11)

    # classifier

    x_class_2 = Conv2D(filters=2, kernel_size=(1, 1), padding='same', activation='softmax',  # + 1 for the background class
                       kernel_initializer='uniform', name="out_class")(x_class_1)

    #regressor
    x_reg_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='normal', name='conv12_reg')(f11)
    # x_reg_1_0 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='normal')(x_reg_1)

    x_reg_2 = Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])
    # model = Model(input_layer, [x_class_2])

    return model