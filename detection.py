from email import header

import torch
from tool.torch_utils import *
from tool.utils import *
from models.py2_Yolov4_model import Yolov4
import os
from multiprocessing import Process

import cv2
import time
import copy
import argparse
import math
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from std_msgs.msg import Float32MultiArray 
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Transform
from std_msgs.msg import Bool

from API.tracker import Tracker
from API.drawer import Drawer
from calibration.calib import Calibration
from dist import Distance


class YOLOv4_Det:
    ### Need to change image_shape 
    def __init__(self, args, image_shape):
        camera_path = ['calibration/f_camera_1280.txt', 'calibration/l_camera_1280.txt', 'calibration/r_camera_1280.txt']
        imu_camera_path = './calibration/camera_imu.txt'
        LiDAR_camera_path = 'calibration/f_camera_lidar_1280.txt'

        self.calib = Calibration(camera_path, imu_camera_path , LiDAR_camera_path)
        self.est_dist = Distance(self.calib)

        self.args = args
        self.img_shape = image_shape
        self.YOLOv4 = self.LoadModel()

        self.tracker = Tracker((self.img_shape[1], self.img_shape[0]), min_hits=1, num_classes=args.n_classes, interval=args.interval)
        self.drawer = Drawer(args.namesfile)  
        self.bridge = CvBridge()

        self.cur_front_img = {'img':None, 'header':None}
        self.cur_left_img = {'img':None, 'header':None}
        self.cur_right_img = {'img':None, 'header':None}
        self.sum_img = {'img':None, 'header':None}

        self.GT_dist = None
        self.cur_pose = None

        rospy.init_node('detection1')

        rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)
        rospy.Subscriber('/Camera/Transform', Transform, self.CamPosecallback, queue_size=10)
        rospy.Subscriber('/Imu/Transform', Transform, self.ImuPosecallback, queue_size=10)

        self.pub_img = rospy.Publisher('/detection/result_front', Image, queue_size=1)
        self.pub_dist_imu = rospy.Publisher("/detection/dist_imu", Float32MultiArray, queue_size=10)
        self.pub_dist = rospy.Publisher("/detection/dist", Float32MultiArray, queue_size=10)
        self.pub_ori_dist = rospy.Publisher("/detection/ori_dist", Float32MultiArray, queue_size=10)

        self.pub_2d_bbox = rospy.Publisher('/detection/bbox_2d', Float32MultiArray, queue_size=1)
        self.pub_3d_bbox = rospy.Publisher('/detection/bbox_3d', Float32MultiArray, queue_size=1)

        rospy.Subscriber('/compressed_image', CompressedImage, self.Front_IMGcallback)
        rospy.Subscriber('/detection/imu', Imu, self.IMUcallback, queue_size=10)

        self.get_new_IMG_msg1 = False
        self.get_new_IMG_msg2 = False
        self.get_new_IMG_msg3 = False
        self.get_new_IMU_msg = True

        self.dist_arr = []
        self.dist_arr_imu = []
        self.GT_arr = []
        self.ori_dist_arr = []


    def IMUcallback(self, msg):
        self.get_new_IMU_msg = True

    def Front_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        front_img = cv2.resize(front_img, (self.img_shape))
        
        self.cur_front_img['img'] = self.calib.undistort(front_img, 'front')
        self.cur_front_img['header'] = msg.header
        self.get_new_IMG_msg1 = True

    def Left_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        left_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        left_img = cv2.resize(left_img, (self.img_shape))
        
        self.cur_left_img['img'] = self.calib.undistort(left_img, 'left')
        self.cur_left_img['header'] = msg.header
        self.get_new_IMG_msg2 = True

    def Right_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        right_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        right_img = cv2.resize(right_img, (self.img_shape))
        
        self.cur_right_img['img'] = self.calib.undistort(right_img, 'right')
        self.cur_right_img['header'] = msg.header
        self.get_new_IMG_msg3 = True

    def image_sum(self, left, front, right):
        result_img = np.hstack((left, front))
        result_img = np.hstack((result_img, right))
        self.sum_img['img'] = result_img
        self.sum_img['header'] = self.cur_right_img['header']

        return self.sum_img['img']

    def LiDARcallback(self, msg):
        for marker in msg.markers:
            lidar_x = -marker.pose.position.x
            lidar_y = marker.pose.position.y
            lidar_z = marker.pose.position.z

        if len(msg.markers) > 0:
            diag = math.sqrt(lidar_x**2+lidar_y**2)
            self.GT_dist = math.sqrt(diag**2 + lidar_z**2)

    def get_bbox_array_msg(self, bboxes, labels, header):
        bbox_array_msg = Float32MultiArray()
        bboxes_info = []
        for idx, bbox in enumerate(bboxes):
            left_top_x = bbox[0]
            left_top_y = bbox[1]
            right_bot_x = bbox[2]
            right_bot_y = bbox[3]

            box_info = [idx, left_top_x, left_top_y, right_bot_x, right_bot_y]
            bboxes_info.append(box_info)

        bbox_array_msg.data = sum(bboxes_info, [])

        return self.pub_2d_bbox.publish(bbox_array_msg)

    def get_3d_box_array_msg(self, pts_3d):
        pt_3d_array_msg = Float32MultiArray()
        pt_3d_info = []
        for idx, pt in enumerate(pts_3d):
            Real_X = pt[0]
            Real_Y = pt[1]
            pt_3d_info.append([Real_X, Real_Y])

        pt_3d_array_msg.data = sum(pt_3d_info, [])

        return self.pub_3d_bbox.publish(pt_3d_array_msg)

    def CamPosecallback(self, msg):

        ## optical flow (dy)
        self.cam_pose = [msg.rotation.x, msg.rotation.y, msg.rotation.z]
    
    def ImuPosecallback(self, msg):

        self.imu_pose = [msg.rotation.x, msg.rotation.y, msg.rotation.z]

    def get_dist_array_msg(self, dist_arr):
        dist_array_msg = Float32MultiArray()
        
        for dist in dist_arr:
            dist_array_msg.data.append(dist[0])

        return dist_array_msg

    def LoadModel(self):
        model = Yolov4(n_classes=self.args.n_classes,pre_weight = args.weightfile, inference=True)
        torch.cuda.set_device(self.args.gpu_num)
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True
        print ('Current cuda device : %s'%(torch.cuda.current_device()))
        return model

    def main(self):
        try:
            new_dist_flag = False
            moving_tra, moving_det = 0., 0.
            frame_ind = 0
            ground_plane = np.array(self.calib.src_pt, np.int32)
            
            while not rospy.is_shutdown():
                if  self.get_new_IMG_msg1  :

                    start = time.time()
                    dets_arr, labels_arr, is_dect = None, None, None

                    if np.mod(frame_ind, self.args.interval) == 0:
                        self.sum_img['img'] = self.cur_front_img['img']
                        orig_im = copy.copy(self.sum_img['img'])
                        orig_im = cv2.resize(orig_im, (self.img_shape))

                        img = cv2.resize(self.sum_img['img'], (320, 320))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        bbox = do_detect(self.YOLOv4, img, 0.2, 0.4)[0]

                        if len(bbox) > 0 and self.cam_pose is not None: 
                            bbox = np.vstack(bbox)
                            output = copy.copy(bbox)
                            output[:,0] = (bbox[:,0] - bbox[:,2] / 2.0) * self.img_shape[0]
                            output[:,1] = (bbox[:,1] - bbox[:,3] / 2.0) * self.img_shape[1]
                            output[:,2] = (bbox[:,0] + bbox[:,2] / 2.0) * self.img_shape[0]
                            output[:,3] = (bbox[:,1] + bbox[:,3] / 2.0) * self.img_shape[1]

                            dets_arr, labels_arr = output[:,0:4], output[:,-1].astype(int)
                            #####
                            task = 'front'
                            self.ori_dist_arr, self.dist_arr, self.dist_arr_imu, pts_3d, ground_plane, new_dist_flag = self.est_dist.getDistance(dets_arr, self.cam_pose, self.imu_pose, task)

                            ## publish for logging in pose_estimation
                            self.pub_ori_dist.publish(self.get_dist_array_msg(self.ori_dist_arr))
                            self.pub_dist.publish(self.get_dist_array_msg(self.dist_arr))
                            self.pub_dist_imu.publish(self.get_dist_array_msg(self.dist_arr_imu))

                            ## Compare for LiDAR
                            self.get_3d_box_array_msg(pts_3d)
                        else:
                            dets_arr, labels_arr = np.array([]), np.array([])
                        is_dect = True

                    elif np.mod(frame_ind, args.interval) != 0:
                        dets_arr, labels_arr = np.array([]), np.array([])
                        is_dect = False

                    pt_det = (time.time() - start)
                    
                    ## draw standard point
                    if len(self.est_dist.center_pts) > 0:
                        for pt in self.est_dist.center_pts:
                            orig_im = cv2.line(orig_im, (int(pt[0]), int(pt[1])),(int(pt[0]), int(pt[1])), (255,255,255), 10)
                        self.est_dist.center_pts= []
                    if frame_ind != 0:
                        moving_det = (frame_ind / float(frame_ind + 1) * moving_det) + (1. / float(frame_ind + 1) * pt_det)
                    
                    show_frame = self.drawer.draw(orig_im, dets_arr, labels_arr, self.dist_arr, self.ori_dist_arr, self.GT_dist, (1. / (moving_det + 1e-8)), is_tracker=False)
                
                    if new_dist_flag:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,0,255), thickness=2)
                    else:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,255,0), thickness=2)
                    
                    if self.pub_img.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = self.bridge.cv2_to_imgmsg(show_frame, "bgr8")
                            # msg.header = self.sum_img['header']
                            self.cur_front_img['header'] = msg.header

                        except CvBridgeError as e:
                            print(e)
                        self.pub_img.publish(msg)

                    frame_ind += 1
                    self.get_new_IMG_msg = False
                    self.get_new_IMG_msg1 = False
                    self.get_new_IMG_msg2 = False
                    self.get_new_IMG_msg3 = False
                    self.get_new_IMU_msg = False
                    new_dist_flag = False
            
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weightfile', default="./weight/0811_1_epoch261.pth")  

    parser.add_argument('--n_classes', default=80, help="Number of classes")
    parser.add_argument('--namesfile', default="data/coco.names", help="Label name of classes")
    parser.add_argument('--gpu_num', default=0, help="Use number gpu")
    parser.add_argument('--interval', default=1, help="Tracking interval")
    args = parser.parse_args()

    image_shape=(1280, 806)

    Detection = YOLOv4_Det(args, image_shape)
    Detection.main()
