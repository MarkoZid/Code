"""
Authors: DIPteam (https://dipteam.feit.ukim.edu.mk/)
E-mail: dipteam42@gmail.com
Course: Letna skola za multimediski tehnologii 2023 - Diogen 2.0
Date: 25.09.2023

Description: load test images, load trained model and use model to evaluate test images
Python version: 3.6


majkati
"""

# python imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.optimizers import Adam

# custom package imports
from Helpers import helper_model
from Helpers import helper_data
from Helpers import helper_losses
from Helpers import helper_tools
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# --- paths ---
version = 'v1'

# NOTE: specify destination paths
srcImagesPath = 'Day2\\Images\\c025'

srcModelPath = 'Models'

dstResultsPath = 'Results'

srcCoeffsPath=os.path.join(srcModelPath, 'regCoeffs')
# create folders to save data from the current execution
resultsPath = os.path.join(dstResultsPath, version)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# --- variables ---
image_extension = '.jpg'

image_size_in = (1920, 1080)     # (columns, rows)
image_size_out = (960, 540)     # (columns, rows)

image_depth = 3   # 1 - grayscale, 3 - color




# img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

num_classes = 1
min_obj_height = 25  # minimum object height in pixels

prob_thr = 0.5  # probability threshold
plot_color = (0, 255, 0)    # color of resulting bounding boxes


# --- load and format data ---
# load full dataset into memory - image data and labels
x_test = helper_data.load_test_data_pretrained_model(im_path=srcImagesPath, im_ext=image_extension,
                                                           im_size_in=image_size_in,
                                                           im_size_out=image_size_out,
                                                           im_depth=image_depth)

print(f'Number of test samples: {x_test.shape[0]}')

# for im in x_test:
#     cv2.imshow('slika', im)
#     cv2.waitKey(0)
# --- prepare ground truth data in required format ---
anchor_stride = 8  # NOTE: depends on the model configuration

# --- construct model ---
lr = 0.0001
model = helper_model.load_model(model_path=os.path.join(srcModelPath, 'model.json'),
                                weights_path=os.path.join(srcModelPath, 'model.h5'))   # build model architecture

# compile model
model.compile(loss={
                  'out_class': helper_losses.rpn_loss_cls_new,
                  'out_reg': helper_losses.rpn_loss_reg,
                   },
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])

# --- convert test images to grayscale (CNN input format) ---
x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test]
x_test_1 = np.array(x_test_1)
x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))

# --- apply model to test data ---
[output_cls, output_reg] = model.predict(x_test_1, verbose=1)
print(output_cls)
print(output_reg)

# --- histogram of predicted probabilities ---
output_cls_pos_flat = output_cls[:, :, :, :-1].flatten()    # select objectness maps for positive objects only
plt.hist(output_cls_pos_flat, density=False, bins=100)  # density=False shows counts, True shows density
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.ylabel('Count')
plt.xlabel('Probability values')
plt.show()


# # --- save results ---
# f = open(os.path.join(srcCoeffsPath, 'norm_coef.txt'), 'r')
# file_contents = f.readlines()
# reg_norm_coef = float(file_contents[0])
#
#
# helper_data.save_results_anchorless_limits(resultsPath, x_test, plot_color, output_cls, output_reg, anchor_stride,
#                                            prob_thr, reg_norm_coef)

# --- save results ---

reg_norm_coef = helper_tools.loadnormcoeffvalues(srcCoeffsPath)

print(reg_norm_coef)

helper_data.save_results_anchorless_limits(resultsPath, x_test, plot_color, output_cls, output_reg, anchor_stride,
                                           prob_thr, reg_norm_coef)
