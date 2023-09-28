import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def line_segments_intersect(x1, x2, y1, y2):
    """
    calculate intersection over union for two 1-d segments
    # Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
    # can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
    # similarly for the ys.
    :param x1: min_point of first segment
    :param x2: max_point of first segment
    :param y1: min_point of second segment
    :param y2: max_point of second segment
    :return: IoU value
    """

    if x2 >= y1 and y2 >= x1:
        # the segments overlap
        intersection = min(x2, y2) - max(y1, x1)
        union = max(x2, y2) - min(x1, y1)
        iou = float(intersection) / float(union)

        return iou

    return 0

def calc_iou_partwise(box1, box2):
    """
    calculate intersection over union for height and width separately
    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou_height: iou value by height [float]
             iou_width: iou value by width [float]
    """

    iou_height = line_segments_intersect(box1[0], box1[2], box2[0], box2[2])
    iou_width = line_segments_intersect(box1[1], box1[3], box2[1], box2[3])

    return iou_height, iou_width

def load_images_annotations(src_path_data, im_size):
    images=[]
    all_boxes=[]
    sceni=os.listdir(src_path_data)
    for scena in sceni:
        path_images_scene=os.path.join(src_path_data,scena,'Images')
        path_annot_scene=os.path.join(src_path_data,scena,'Annotations')

        cams=os.listdir(path_images_scene)

        for cam in cams:
            path_images_cam=os.path.join(path_images_scene,cam)
            path_annot_cam=os.path.join(path_annot_scene,cam)
            im_filenames=os.listdir(path_images_cam)

            for im_filename in im_filenames:
                objects=[]
                image=cv2.imread(os.path.join(path_images_cam,im_filename),0)

                HScale=im_size[0]/image.shape[0]
                WScale=im_size[1]/image.shape[1]

                resized=cv2.resize(image, (im_size[1],im_size[0]), interpolation=cv2.INTER_AREA)

                annot_file=im_filename[:-4]+'.txt'

                bboxes = np.loadtxt(os.path.join(path_annot_cam, annot_file), delimiter=',', ndmin=2).astype(np.int64)
                for bbox in bboxes:
                    object = [min(int(bbox[0] * HScale), resized.shape[0]), min(int(bbox[1] * WScale), resized.shape[1]),
                              min(int(bbox[2] * HScale), resized.shape[0]), min(int(bbox[3] * WScale), resized.shape[1]), 1]
                    objects.append(object)
                    # cv2.rectangle(resized,(object[1],object[0]),(object[3],object[2]),(255,0,0),2)

                # cv2.imshow("resized",resized)
                # cv2.waitKey(0)

                resized = resized / 255
                resized = resized.reshape(resized.shape[0], resized.shape[1], 1)
                images.append(resized)
                all_boxes.append(objects)

    images=np.array(images)
    all_boxes=np.array(all_boxes)

    return  images, all_boxes




def is_small(img_dims, small_dims, min_h):

    if (img_dims[0] > small_dims['h'] and img_dims[1] > small_dims['w']) or img_dims[0] > min_h:
        return False
    else:
        return True


def load_data(scenes, scene_path, start_frame, stop_frame, sekoja_x_ramka, dest_path_annot,
                          small_dims, min_h):

    print("divideavatarsbyfolder: ")

    # for scene in os.listdir(scene_path):
    for scene in scenes:

        path_tmp = os.path.join(dest_path_annot, scene)
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)

        print(scene,path_tmp)
        for cam in os.listdir(os.path.join(scene_path, scene)):
            if "map" in cam:
                continue

            dst_dir_im = os.path.join(dest_path_annot, scene,'Images', cam)
            dst_dir_ann=os.path.join(dest_path_annot, scene,'Annotations', cam)
            if not os.path.exists(dst_dir_im):
                os.makedirs(dst_dir_im)

            if not os.path.exists(dst_dir_ann):
                os.makedirs(dst_dir_ann)
            print(cam,dst_dir_ann)
            vid_cap = cv2.VideoCapture(os.path.join(scene_path, scene, cam, "video.mp4"))  # create video capture object
            txt_path = os.path.join(scene_path, scene, cam, "label.txt")
            labels = pd.read_csv(txt_path, delimiter=',',
                                 names=["fn", "id", "x", "y", "w", "h", "xr", "yr", "wr", "hr"])
            # print(labels.head())
            success, frame = vid_cap.read()  # load first frame, populate success variable

            # extract frames
            frame_cnt = 0  # frame counter

            annot_path_dir = dst_dir_ann
            im_path_dir = dst_dir_im
            if not os.path.exists(annot_path_dir):
                os.makedirs(annot_path_dir, exist_ok=True)

            if not os.path.exists(im_path_dir):
                os.makedirs(im_path_dir, exist_ok=True)
            while success:

                if frame_cnt < start_frame:
                    success, frame = vid_cap.read()  # load first frame, populate success variable
                    frame_cnt += 1
                    continue

                if stop_frame is not None:
                    if frame_cnt == stop_frame:
                        break

                if frame_cnt % sekoja_x_ramka == 0:
                    # cv2.imshow("a", frame)
                    # cv2.waitKey(0)
                    pairs=[]
                    labels_by_frame = labels[labels["fn"] == frame_cnt]
                    # print(labels_by_frame.head())


                    for index, row in labels_by_frame.iterrows():
                        annot=[row["y"],row["x"],  row["y"] + row["h"], row["x"] + row["w"]]


                        if not is_small([row['h'], row['w']], small_dims, min_h):
                            pairs.append(annot)
                    if len(pairs)!=0:

                        cv2.imwrite(os.path.join(im_path_dir, str(frame_cnt).zfill(6) + ".jpg"), frame)
                        np.savetxt(os.path.join(annot_path_dir, str(frame_cnt).zfill(6) + ".txt"), pairs, delimiter=',', fmt='%i')

                success, frame = vid_cap.read()  # load first frame, populate success variable
                frame_cnt += 1




def generate_anchor_level_object_masks(bboxes, img_dims, anchor_stride):
    """
    generate masks of bounding boxes in the output-level matrices
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image !!!!!!
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :return: object_masks - anchor-level masks of object locations [ndarray]
    """

    object_masks_dims = (np.int64(img_dims[0] / anchor_stride), np.int64(img_dims[1] / anchor_stride))
    object_masks = np.zeros(object_masks_dims).astype(np.int64)

    # iterate over ground truth files for each image

    for bbox in bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2
        if bbox[-1] == 6:
            continue
        # calculate bbox coordinates in output matrix
        # NOTE: floor & ceil namesto round
        bbox_out = [max(0, np.int64(np.floor(bbox[0] / anchor_stride))),  # min_row
                    max(0, np.int64(np.floor(bbox[1] / anchor_stride))),  # min_col
                    min(object_masks_dims[0] - 1, np.int64(np.ceil(bbox[2] / anchor_stride))),  # max_row
                    min(object_masks_dims[1] - 1, np.int64(np.ceil(bbox[3] / anchor_stride)))]  # max_col

        # fill object mask
        object_masks[bbox_out[0]:bbox_out[2], bbox_out[1]:bbox_out[3]] = 1

    return object_masks



def create_gt_parallel(bboxes, obj_mask, img_dims, anchor_stride, iou_low_b, iou_high_b, iou_low_s, iou_high_s, num_negs_ratio,
                                                                    num_classes, negative_mask, negative_mask_racni):
    """
    generate ground truth output for fully convolutional network for object detection
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param obj_mask: mask of possible object locations (bounding box centers which lie within an object bounding box) [ndarray]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_cls_arr - ground truth classes [ndarray]
             output_cls_arr - ground truth regression (normalized to [-1, 1]) [ndarray]
             valid_inds - indices of images containing at least one object [list]
             reg_norm_coef - normalization coefficient for ground truth regression data [float]
    """
    # print('bboxes')
    # print(bboxes)
    # print(x)

    # print(img_dims[0])
    # print(img_dims[1])

    output_dims_class = (np.int64(img_dims[0] / anchor_stride), np.int64(img_dims[1] / anchor_stride), num_classes + 1)
    output_dims_reg = (np.int64(img_dims[0] / anchor_stride), np.int64(img_dims[1] / anchor_stride), 4)
    # print(output_dims_class)

    output_class = np.zeros(output_dims_class).astype(np.int64)
    output_reg = np.zeros(output_dims_reg).astype(np.float64)

    # first position of an anchor center
    start_r = np.int64(np.round(anchor_stride / 2))
    start_c = np.int64(np.round(anchor_stride / 2))

    [r, c] = np.where(obj_mask == 1)  # possible object locations
    iou_low_between = 0.1
    iou_high_between = iou_high_b
    # ako nema pozitivni objekti
    # da se proveri dali ima 6ki
    # ako da, da se popolnat negativnite
    # ako ne, return none

    # print('DEBUG: ')
    # print(len(r))
    # print(np.sum(negative_mask_racni))
    # print(bboxes)
    bboxes_num_pos = [0] * len(bboxes)

    if len(r) == 0:  # nema pozitivni

        if np.sum(negative_mask_racni) > 0:
            print('tukaaaaaaaaaaa')  # ima 6ki, popolni negativni
            [r_negs, c_negs] = np.where(negative_mask_racni[:, :] == 1)
            for i in range(len(r_negs)):
                if np.sum(output_class[r_negs[i], c_negs[i], :-1]) == 0:
                    output_class[r_negs[i], c_negs[i], -1] = 1

            return output_class, output_reg
            # return output_class, output_reg,bboxes_num_pos

        else:
            # nema nisto
            print('ovde kaj non1')
            return None, None

    # ctr = 0
    # for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
    #     for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers
    #         ctr += 1
    #
    # print(f'Iterations: {len(r) ** 2} {ctr}')

    for object_num in range(len(r)):
        output_row = r[object_num]
        output_col = c[object_num]

        for ind, bbox in enumerate(bboxes):  # iterate through annotated bounding
            # print(f'Bbox: {bbox}')

            iou_high = iou_high_b
            iou_low = iou_low_b

            flag_zanigde = False

            # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
            bbox_h = bbox[2] - bbox[0]
            bbox_w = bbox[3] - bbox[1]
            bbox_class = bbox[4]
            # print(type(bbox_class))
            if bbox_h <= 1 or bbox_w <= 1:
                continue
            if bbox_class == 6:
                continue
            # NOTE ova da se vrati ako treba(ako snema mali)

            # if bbox_h < 30:
            #     iou_high = iou_high_s
            #     iou_low = iou_low_s

            # if bbox[1] < 5 or bbox[2] > img_dims[0] - 5 or bbox[3]>img_dims[1]-5:
            #     # treba dvojka vo dvete klasi
            #     flag_zanigde = True

            half_bbox_dim_h = np.int64(np.round(bbox_h / 2))
            half_bbox_dim_w = np.int64(np.round(bbox_w / 2))

            center_row = output_row * anchor_stride + start_r
            center_col = output_col * anchor_stride + start_c

            anchor = [max(0, center_row - half_bbox_dim_h),
                      max(0, center_col - half_bbox_dim_w),
                      min(center_row + half_bbox_dim_h, img_dims[0]),
                      min(center_col + half_bbox_dim_w, img_dims[1])]
            # min_row, min_col, max_row, max_col

            # iou = helper_postprocessing.calc_iou(bbox, anchor)
            iou_h, iou_w = calc_iou_partwise(bbox, anchor)
            # print(iou_w)

            # print(f'IoU_H {iou_h} IoU_W {iou_w} Bbox {bbox} Anchor {anchor}')
            if iou_h >= iou_high and iou_w >= iou_high:  # pozitivnite
                bboxes_num_pos[ind] += 1
                # print(f'POS: IOU: {iou_h} {iou_w} \t {bbox} \t {anchor}')
                # if iou >= iou_high:   # pozitivnite
                # print(iou_h, iou_w)
                # positive sample: set class, calculate deltas

                if flag_zanigde:
                    output_class[output_row, output_col, bbox_class - 1] = 2
                else:
                    output_class[output_row, output_col, bbox_class - 1] = 1

                    # set deltas - current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

            if ((iou_h < iou_high_between) and (iou_h > iou_low_between)) and ((iou_w < iou_high_between) and (iou_w > iou_low_between)):
                # print(f'INBETWEEN: IOU: {iou_h} {iou_w} \t {bbox} \t {anchor}')

                # if iou < iou_high and iou > iou_low:
                # IOU between iou_min and iou_max
                # print(output_row, output_col, bbox_class-1)
                output_class[output_row, output_col, bbox_class - 1] = 2  # temporarily mark class with 2

                if not flag_zanigde:
                    # set deltas
                    # current location minus correct location
                    delta_r = bbox[0] - anchor[0]
                    delta_c = bbox[1] - anchor[1]

                    h_percent = bbox_h / img_dims[0]
                    w_percent = bbox_w / img_dims[1]

                    output_reg[output_row, output_col, 0] = delta_r
                    output_reg[output_row, output_col, 1] = delta_c
                    output_reg[output_row, output_col, 2] = h_percent
                    output_reg[output_row, output_col, 3] = w_percent

    #dilate output_class

    # kernel=np.ones((3,3), np.uint8)
    # k=output_class[:, :, 0]
    # output_class=output_class.astype(np.uint8)
    # for ind in range(0,num_classes):
    #     output_class[:,:,ind]=cv2.dilate(output_class[:,:,ind],kernel)


    # mark negative samples

    for out_row in range(output_class.shape[0]):  # iterate through rows of output
        for out_col in range(output_class.shape[1]):  # iterate through columns of output

            if sum(output_class[out_row, out_col, :]) == 0:  # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                output_class[out_row, out_col, num_classes] = 1

    # replace 2s with 0s
    output_class = np.where(output_class == 2, 0, output_class)

    # --- select negative samples ---
    # count positives and negatives
    num_positives = np.sum(output_class[:, :, 0:num_classes])

    # find negatives
    negs = output_class[:, :, num_classes]
    [r, c] = np.where(negs == 1)

    # select negatives to remove
    ind_to_remove = np.arange(len(r))
    np.random.shuffle(ind_to_remove)

    num_neg = min(len(r), num_positives * num_negs_ratio)  # number of positive to negative samples ratio: 1 to 10
    num_to_remove = len(r) - num_neg
    ind_to_remove = ind_to_remove[:num_to_remove]

    # remove negatives that were not selected
    for ind in ind_to_remove:
        output_class[r[ind], c[ind], :] = 0

    for ind in ind_to_remove:
        output_reg[r[ind], c[ind], :] = 0

    if negative_mask is not None:
        [r, c] = np.where(negative_mask[:, :] == 1)
        for i in range(len(r)):
            if (np.sum(output_class[r[i], c[i], :-1]) == 0):
                output_class[r[i], c[i], -1] = 1

        # output_class[:,:,-1] = np.where(negative_mask == 1, 1, output_class[:,:,-1])

    if np.sum(negative_mask_racni) > 0:
        [r, c] = np.where(negative_mask_racni[:, :] == 1)
        for i in range(len(r)):
            if (np.sum(output_class[r[i], c[i], :-1]) == 0):
                output_class[r[i], c[i], -1] = 1

    # num_negatives = np.sum(output_class[:, :, -1])

    if num_positives > 0:
        # if np.sum(output_class) > 0:

        # NOTE da se vrati

        # return output_class, output_reg, bboxes_num_pos
        # print(output_class.shape)
        return output_class, output_reg
    else:
        print('ovde kaj non2')
        return None, None

def load_test_data_pretrained_model(im_path, im_ext, im_size_in, im_size_out, im_depth):
    """
    load and pre-process image data (resize, normalize, adjust number of channels)
    if image size is changed, adapt annotations
    :param im_path: global path of folder containing images [string]
    :param im_ext: file extension of images (ex. ".bmp") [string]
    :param im_size_in: dimensions of original images (cols, rows) [tuple]
    :param im_size_out: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 - greyscale or 3 - RGB) [int]
    :return: images_list - array of normalized depth maps [ndarray]
    """

    images_list = []       # array of normalized images

    # --- list images in source folder ---
    for im_name in os.listdir(os.path.join(im_path)):

        # --- load image ---

        if im_name[-4:] != im_ext:  # exclude non-image files
            continue

        frame_num = int(im_name.split('.')[0].lstrip('0'))

        if im_depth == 3:   # select color mode: greyscale or RGB
            image = cv2.imread(os.path.join(im_path, im_name))
        else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)

        if im_size_out != im_size_in:     # resize images
            image = cv2.resize(image, im_size_out, interpolation=cv2.INTER_AREA)

        image = image.reshape(image.shape[0], image.shape[1], im_depth)

        images_list.append(image)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    images = np.array(images_list)



    return images



def save_results_anchorless_limits(results_path, images, plot_color, output_cls, output_reg, anchor_stride, prob_thr, norm_coef):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int64(np.round(anchor_stride / 2))
    start_c = np.int64(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):

        res = output_cls[im_ind, :, :, 0]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int64(center_row - np.round(h / 2))
            min_col = np.int64(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int64(min_row + delta_r)
            min_col_adj = np.int64(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int64(min_col_adj))
            min_row_adj = max(0, np.int64(min_row_adj))
            max_col_adj = min(np.int64(max_col_adj), img_dims[1])
            max_row_adj = min(np.int64(max_row_adj), img_dims[0])

            # cv2.circle(image, (center_col, center_row), 3, color=plot_color, thickness=3)     # plot object centers
            cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        print (results_path)
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)


def save_results_anchorless_limits_cls(results_path, images, plot_color, output_cls, output_reg, anchor_stride, prob_thr, norm_coef):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int64(np.round(anchor_stride / 2))
    start_c = np.int64(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):

        res = output_cls[im_ind, :, :, 0]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int64(center_row - np.round(h / 2))
            min_col = np.int64(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int64(min_row + delta_r)
            min_col_adj = np.int64(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int64(min_col_adj))
            min_row_adj = max(0, np.int64(min_row_adj))
            max_col_adj = min(np.int64(max_col_adj), img_dims[1])
            max_row_adj = min(np.int64(max_row_adj), img_dims[0])

            max_col = np.int64(min(min_col + w, img_dims[1]))
            max_row = np.int64(min(min_row + h, img_dims[0]))
            min_col = np.int64(max(0, min_col))
            min_row = np.int64(max(0, min_row))

            # cv2.circle(image, (center_col, center_row), 3, color=plot_color, thickness=3)     # plot object centers
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)
            cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
