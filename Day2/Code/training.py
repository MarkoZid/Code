'''
IMPORT LIBRARIES
'''

from Helpers import helper_data, helper_model, helper_losses, construct_model, helper_stats, \
    visualize  # custom libraries in project
import os
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''
SET ROOT PATHS FOR SOURCE AND DESTINATION
'''

OriginalModelsPath = r'C:\Users\tatja\Desktop\LETNASKOLA\Models'  # path to pretrained model
rootDestPath = r'C:\Users\tatja\Desktop\LETNASKOLA\MachineVision'  # path to folders for results
version = "v1"  # new model version

models_dest_path = os.path.join(rootDestPath, "Models", version)  # destination path for new model

# create destination path for models
if not os.path.exists(models_dest_path):
    os.makedirs(models_dest_path)

''''
LOAD MODEL ARCHITECTURE AND WEIGHTS
'''

# load model
model = helper_model.load_model(model_path=os.path.join(OriginalModelsPath, 'model.json'),
                                weights_path=os.path.join(OriginalModelsPath, 'model.h5'))

# print model summary (model architecture will be shown)
model.summary()

'''
LOADING VIDEOS FOR TRAINING AND VALIDATION AND EXTRACTING IMAGES AND ANNOTATIONS AND SAVING THEM IN DESTINATION FOLDER
'''

trening_val = ["train", "validation"]
dataset_path = r'C:\Users\tatja\Desktop\LETNASKOLA\Dataset'  # path to Dataset

scenes_train = ["S002"]  # chosen scenes for training
scenes_val = ["S005"]  # chosen scenes for validation
dest_path_data = r'C:\Users\tatja\Desktop\LETNASKOLA\Novo'  # destination folder

if not os.path.exists(dest_path_data):
    os.makedirs(dest_path_data, exist_ok=True)

frame_to_start = 0 * 60 * 30  # None - whole video, (int) - num of stop frame
frame_to_stop = 3 * 60 * 30  # 1001  # None - whole video, (int) - num of stop frame
sekoja_x_ramka = 5

# exclude small images from training
small_dims = {'h': 10, 'w': 5}
min_h = 10

# iterating over train val and test folders
for ind, fl in enumerate(trening_val):

    dest_path_data_by_fld = os.path.join(dest_path_data, fl)  # destination path by folder
    if not os.path.exists(dest_path_data_by_fld):
        os.makedirs(dest_path_data_by_fld)

    scene_path = os.path.join(dataset_path, fl)  # destination_path by scene

    # scenes according to folder name
    if fl == "train":
        scenes = scenes_train
    else:
        scenes = scenes_val

    # load data
    helper_data.load_data(scenes, scene_path, frame_to_start, frame_to_stop,
                          sekoja_x_ramka, dest_path_data_by_fld,
                          small_dims, min_h)

'''

LOADING IMAGES AND MAKING GROUND TRUTH

'''
src_path_train = r'C:\Users\tatja\Desktop\LETNASKOLA\Novo\train'  # source path of extracted images and annotations for training
src_path_val = r'C:\Users\tatja\Desktop\LETNASKOLA\Novo\validation'  # source path of extracted images and annotations for validation

# imgDims = {'rows': 281, 'cols': 469}
imgDims = {'rows': 540, 'cols': 960}  # dimensions of image
num_classes = 1  # person(1 class)
img_depth = 1

img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

im_size = (img_dims[0], img_dims[1])  # model input size

anchor_stride = 8

# IOU thresholds for selecting positive and negative anchors
iou_low = 0.4
iou_high = 0.55
iou_low_s = 0.4
iou_high_s = 0.55
num_negs_ratio = 6

# input data for model
out_class_train = []
out_reg_train = []
out_class_val = []
out_reg_val = []

# load images and annotations
images_train_all, bboxes_train_all = helper_data.load_images_annotations(src_path_train, im_size)
images_val_all, bboxes_val_all = helper_data.load_images_annotations(src_path_val, im_size)

# create mask for locations of negative anchors - OPTIONAL***
negative_mask = None
object_masks_dims = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
negative_mask_racni = np.zeros(object_masks_dims).astype(np.int)

for ind, image in enumerate(images_train_all):
    bboxes_train = bboxes_train_all[ind]
    obj_masks_train = helper_data.generate_anchor_level_object_masks(bboxes_train, img_dims, anchor_stride)

    out_class_train_per_img, out_reg_train_per_img = helper_data.create_gt_parallel(bboxes_train, obj_masks_train[:, :],
                                                                                    img_dims, anchor_stride,
                                                                                    iou_low,
                                                                                    iou_high, iou_low_s, iou_high_s,
                                                                                    num_negs_ratio, num_classes,
                                                                                    negative_mask, negative_mask_racni)

    out_class_train.append(out_class_train_per_img)
    out_reg_train.append(out_reg_train_per_img)

# visualize ground truth for training
# visualize.visualize_gt(out_class_train, out_reg_train, images_train_all, anchor_stride)


for ind, image in enumerate(images_val_all):
    bboxes_val = bboxes_val_all[ind]
    obj_masks_val = helper_data.generate_anchor_level_object_masks(bboxes_val, img_dims, anchor_stride)

    out_class_val_per_img, out_reg_val_per_img = helper_data.create_gt_parallel(bboxes_val, obj_masks_val[:, :],
                                                                                img_dims, anchor_stride,
                                                                                iou_low,
                                                                                iou_high, iou_low_s, iou_high_s,
                                                                                num_negs_ratio, num_classes,
                                                                                negative_mask, negative_mask_racni)
    out_class_val.append(out_class_val_per_img)
    out_reg_val.append(out_reg_val_per_img)

# visualize ground truth for validation
# visualize.visualize_gt(out_class_val, out_reg_val, images_val_all, anchor_stride)

out_class_train = np.asarray(out_class_train)
out_reg_train = np.asarray(out_reg_train)
out_class_val = np.asarray(out_class_val)
out_reg_val = np.asarray(out_reg_val)

'''PREPARING DATA FOR TRAINING (NORMALIZING REGRESSOR OUTPUT'''

file_path_reg_coef = os.path.join(models_dest_path, "regCoeffs")
if not os.path.exists(file_path_reg_coef):
    os.mkdir(file_path_reg_coef)

reg_norm_coef_position_rows = np.max(np.abs(out_reg_train[:, :, :, 0]))
filename = 'reg_norm_coef_position_rows.txt'
fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
pickle.dump(reg_norm_coef_position_rows, fid1_1)
fid1_1.close()

reg_norm_coef_position_cols = np.max(np.abs(out_reg_train[:, :, :, 1]))
filename = 'reg_norm_coef_position_cols.txt'
fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
pickle.dump(reg_norm_coef_position_cols, fid1_1)
fid1_1.close()

# size
reg_norm_coef_size_height = np.max(np.abs(out_reg_train[:, :, :, 2]))
filename = 'reg_norm_coef_size_height.txt'
fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
pickle.dump(reg_norm_coef_size_height, fid1_2)
fid1_2.close()

reg_norm_coef_size_width = np.max(np.abs(out_reg_train[:, :, :, 3]))
filename = 'reg_norm_coef_size_width.txt'
fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
pickle.dump(reg_norm_coef_size_width, fid1_2)
fid1_2.close()

# normalize training data
out_reg_norm = out_reg_train

print(f'Koeficient pozicii: {reg_norm_coef_position_rows} {reg_norm_coef_position_cols}')
print(f'Koeficient dimenzii: {reg_norm_coef_size_height} {reg_norm_coef_size_width}')

# normalize train data
out_reg_norm[:, :, :, 0] = out_reg_norm[:, :, :, 0] / reg_norm_coef_position_rows
out_reg_norm[:, :, :, 1] = out_reg_norm[:, :, :, 1] / reg_norm_coef_position_cols

out_reg_norm[:, :, :, 2] = out_reg_norm[:, :, :, 2] / reg_norm_coef_size_height
out_reg_norm[:, :, :, 3] = out_reg_norm[:, :, :, 3] / reg_norm_coef_size_width

# normalize validation data
out_reg_val_norm = out_reg_val

out_reg_val_norm[:, :, :, 0] = out_reg_val[:, :, :, 0] / reg_norm_coef_position_rows
out_reg_val_norm[:, :, :, 1] = out_reg_val[:, :, :, 1] / reg_norm_coef_position_cols

out_reg_val_norm[:, :, :, 2] = out_reg_val[:, :, :, 2] / reg_norm_coef_size_height
out_reg_val_norm[:, :, :, 3] = out_reg_val[:, :, :, 3] / reg_norm_coef_size_width

''' PREPARING MODEL '''

lr = 0.001
batch_size = 2
epochs = 3

modelsPath = models_dest_path

model_json = model.to_json()
with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    json_file.write(model_json)

# compile model
model.compile(loss={
    'out_class': helper_losses.rpn_loss_cls_new,
    # 'out_class' and 'out_reg' are the names of the output layers defined when constructing the model architecture
    'out_reg': helper_losses.rpn_loss_reg

},
    optimizer=Adam(lr=lr),
    metrics=['accuracy'])

print(out_class_train.shape)
print(out_class_val.shape)

model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{loss:.4f}.hdf5'),
                                   # epoch number and val accuracy will be part of the weight file name
                                   monitor='loss',  # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=True)  # True saves only the weights after epochs where the monitored value (val accuracy) is improved

history = model.fit(x=images_train_all, y={"out_class": out_class_train, 'out_reg': out_reg_norm},
                    batch_size=batch_size,  # number of samples to process before updating the weights
                    epochs=epochs,
                    callbacks=[model_checkpoint],

                    verbose=1,
                    validation_data=(images_val_all, {"out_class": out_class_val, 'out_reg': out_reg_val_norm}))

print('model fitted')

print(model.summary())  # parameter info for each layer
with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:  # save model summary
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# save model configuration and weights

model_json = model.to_json()  # serialize model architecture to JSON
with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5

plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'),
           show_shapes=True)  # save diagram of model architecture

print('d')

helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)
