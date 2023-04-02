from distutils.errors import DistutilsTemplateError
import cv2
import numpy as np
import os
import math


class Drawer(object):
    def __init__(self, classes_path):
        self.classes_path = classes_path

        self.redColor = (0, 51, 255)
        self.whiteColor = (255, 255, 255)
        self.thickness = 2
        # self.fontSize = 1.5
        self.fontSize = 3

        self.fontFace = cv2.FONT_HERSHEY_TRIPLEX
        self.max_id = 1000

        self._init_params()

    def _init_params(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        self.labels = [c.strip() for c in class_names]
        # self.labels = ["Back", "Car", "Ped", "Cycl", "Motor", "Truck", "Bus", "Van"]

    @staticmethod
    def cal_iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        iou = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (
                bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return iou

    def draw(self, show_img, dets, labels, dists, ori_dist_arr, lidar_dist, fps, is_tracker=True):
        # dets = dets[np.newaxis]
        # labels = labels[np.newaxis]
    
        # dets for detector: x1, y1, x2, y2 , label_memory , distance 
        # dets for tracker: x1, y1, x2, y2, objectID, is_update, labelID
        h, w = show_img.shape[0:2]
        #show_img = img.copy()

        dets[dets < 0.] = 0.  # some kalman predictions results are negative
        dets = dets.astype(np.uint32)
        
        # Draw processing time
        FPS_str = "FPS: {:.1f}".format(fps)
        inform_size = cv2.getTextSize(FPS_str, self.fontFace, self.fontSize, self.thickness)
        margin_h, margin_w = int(0.01 * h), int(0.01 * w)
        bottom_left = (margin_w, inform_size[0][1] + margin_h)
        # cv2.putText(show_img, FPS_str, bottom_left, self.fontFace, self.fontSize, self.redColor, self.thickness)

        for idx in range(dets.shape[0]):
            dist_ct = 0
            offset = 6

            ##RECTANGLE
            cv2.rectangle(show_img, (dets[idx, 0], dets[idx, 1]), (dets[idx, 2], dets[idx, 3]),self.redColor, self.thickness)
            
            # print(dets)
            if is_tracker:
                label_id = dets[idx, 6]
            else:
                label_id = labels[idx]
            label_str = self.labels[label_id]
            label_dist = self.labels

            width, height = dets[idx, 2] - dets[idx, 0], dets[idx, 3] - dets[idx, 1]

            if height < 50:
                label_size = cv2.getTextSize(label_str, self.fontFace, self.fontSize*0.5, self.thickness)
                bottom_left = (dets[idx, 0], dets[idx, 1] + int(0.5 * label_size[0][1]) - offset)
                dist_position = (dets[idx, 0], dets[idx, 3] + offset+5)
                ori_dist_position = (dets[idx, 0], dets[idx, 3] + offset+18)

                                    
                # cv2.rectangle(show_img, (dets[idx, 0], dets[idx, 1] - int(0.5 * label_size[0][1]) - offset),
                #             (dets[idx, 0] + label_size[0][0], dets[idx, 1] + int(0.7 * label_size[0][1]) - offset),
                #             self.redColor, thickness=-1)
                # cv2.putText(show_img, label_str, bottom_left, self.fontFace, self.fontSize*0.5, self.whiteColor,thickness=1)

                if len(dists) > 0 and dists is not None and len(ori_dist_arr) > 0:
                    # for i, dist in enumerate(dists):
                    ##DRAW (DIST)
                    cv2.putText(show_img, str(dists[dist_ct]), dist_position, cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, (0, 255, 0),thickness=2)
                    cv2.putText(show_img, str(ori_dist_arr[dist_ct]), ori_dist_position, cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, (0, 0, 255),thickness=2)
                    dist_ct+=1

            else:
                label_size = cv2.getTextSize(label_str, self.fontFace, self.fontSize, self.thickness)
                bottom_left = (dets[idx, 0], dets[idx, 1] + int(0.5 * label_size[0][1]))
                # dist_position = (dets[idx, 0], dets[idx, 3] + offset+10)
                # ori_dist_position = (dets[idx, 0], dets[idx, 3] + offset+35)

                dist_position = (dets[idx, 0], dets[idx, 3] + offset+30)
                ori_dist_position = (dets[idx, 0], dets[idx, 3] + offset+80)
                # cv2.rectangle(show_img, (dets[idx, 0], dets[idx, 1] - int(0.5 * label_size[0][1])),
                #             (dets[idx, 0] + label_size[0][0], dets[idx, 1] + int(0.7 * label_size[0][1])),
                #             self.redColor, thickness=-1)
                # cv2.putText(show_img, label_str, bottom_left, self.fontFace, self.fontSize, self.whiteColor,thickness=1)
                
                if len(dists) > 0 and dists is not None and len(ori_dist_arr) > 0:
                    ##DRAW (DIST)
                    cv2.putText(show_img, str(dists[dist_ct]), dist_position, cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, (0, 255, 0),thickness=2)
                    cv2.putText(show_img, str(ori_dist_arr[dist_ct]), ori_dist_position, cv2.FONT_HERSHEY_SIMPLEX, self.fontSize*0.5, (0, 0, 255),thickness=2)
                    dist_ct+=1
            # Add object ID
            if is_tracker:
                id_int = np.mod(dets[idx, 4], self.max_id)               
                if height < 50:
                    offset = 7
                    id_size = cv2.getTextSize(str(id_int), self.fontFace, self.fontSize*0.5, self.thickness)
                    bottom_left = (dets[idx, 2] - id_size[0][0], dets[idx, 3] + offset)
                    cv2.rectangle(show_img, (dets[idx, 2] - id_size[0][0], dets[idx, 3] - id_size[0][1] + offset),
                                (dets[idx, 2], dets[idx, 3] + offset), self.redColor, thickness=-1)
                    cv2.putText(show_img, str(id_int), bottom_left, self.fontFace, self.fontSize*0.5, self.whiteColor, thickness=2)
                else:
                    id_size = cv2.getTextSize(str(id_int), self.fontFace, self.fontSize, self.thickness)
                    bottom_left = (dets[idx, 2] - id_size[0][0], dets[idx, 3])
                    cv2.rectangle(show_img, (dets[idx, 2] - id_size[0][0], dets[idx, 3] - id_size[0][1]),
                                (dets[idx, 2], dets[idx, 3]), self.redColor, thickness=-1)
                    cv2.putText(show_img, str(id_int), bottom_left, self.fontFace, self.fontSize, self.whiteColor, thickness=2)

            ## distance
            # if len(dists) > 0 and dists != None and lidar_dist != None:
                # Dist_str = "Dist_%d : %.2f / %.2f"%(dist_ct-1, dists[dist_ct-1][0], lidar_dist)
                # cv2.putText(show_img, Dist_str, (int(0.01 * w), inform_size[0][1] + int(0.01 * h)+(dist_ct-1*25)+50), self.fontFace, self.fontSize*0.5, (0,0,0), self.thickness)

        return show_img


