import numpy as np
import cv2


def visualize_gt(output_cls_all, output_reg_all,images,anchor_stride):

    for ind, image in enumerate(images):
        img_dims = (image.shape[0], image.shape[1])  # height, width
        output_cls=output_cls_all[ind]
        output_reg=output_reg_all[ind]
        num_classes = output_cls.shape[2] - 1

        # calculate location of first (top left) anchor center - start at half of stride size
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))

        res = output_cls[:, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c
            # print(d[pred_ind])
            # print(center_row)
            # print(center_col)
            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[r[pred_ind], c[pred_ind], 1]

            h_percent = output_reg[r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            max_col = np.int(min(min_col + w, img_dims[1]))
            max_row = np.int(min(min_row + h, img_dims[0]))
            min_col = np.int(max(0, min_col))
            min_row = np.int(max(0, min_row))

            if d[pred_ind] == 1:
                cv2.circle(image, (center_col, center_row), 2, color=(255, 0, 0), thickness=3)  # plot object centers
            else:
                cv2.circle(image, (center_col, center_row), 2, color=(0, 255, 0), thickness=3)  # plot object centers

            cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(255, 255, 0), thickness=2)

        cv2.imshow('slika', image)
        cv2.waitKey(0)