import matplotlib.pyplot as plt
import os
import numpy as np

from keras.callbacks import Callback

def save_training_logs_ssd(history, dst_path):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save combined loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('Combined loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'joint_loss.png'))
    plt.close()

    #--- save classification loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['out_class_loss'], 'r')
    plt.plot(history.history['val_out_class_loss'], 'g')
    plt.title('Classification loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_class', 'val_class'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
    plt.close()

    # --- save regression loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['out_reg_loss'], 'r')
    plt.plot(history.history['val_out_reg_loss'], 'g')
    plt.title('Regression loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_reg', 'val_reg'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'regression_loss.png'))
    plt.close()

    # --- save losses of training and validation sets as txt files ---
    joint_losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'joint_loss.txt'), joint_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
