import numpy as np
import cv2
import math
import time

import io, os
import pandas as pd
import tf
import timeit
import rospy

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray

from scipy.spatial.transform import Rotation as sci_R

import matplotlib.pyplot as plt

from geometry_msgs.msg import Transform
from std_msgs.msg import Bool

from calibration.calib import Calibration
import ROS_tools

class PoseEstimation:
    def __init__(self,calibration):

        self.ROS = ROS_tools.ROS(calibration)

        self.cur_img = {'img':None, 'header':None}
        self.get_new_IMG_msg = False
        self.get_new_imu_msg = False
        self.pose_flag = True

        self.preproc_ct = 0
        self.img_cnt = 0.0
        self.init_dy = True

        self.pre_IMU_pitch = 0
        self.max_dy_ori = 0
        self.dy_sum = 0
    
    
    def featureDetection(self, first_frame):

        h, w = first_frame.shape
        roi_image = first_frame[(h/3+50):(2*h/3+100),(w/3-100):(2*w/3+100)] 
        roi_wmask = np.zeros((roi_image.shape[0],w/3-100), np.uint8) 
        roi_image = np.hstack((roi_wmask, roi_image))
        roi_hmask = np.zeros((h/3+50,roi_image.shape[1]), np.uint8)
        roi_image = np.vstack((roi_hmask, roi_image))

        roi_det = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        roi_pre_kp = roi_det.detect(roi_image)

        roi_pre_kp = np.array([x.pt for x in roi_pre_kp], dtype=np.float32)

        return roi_pre_kp

    def featureTracking(self, image_ref, image_cur, ref_kp):

        lk_params = dict(winSize  = (23,23), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        cur_kp, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref_kp, None, **lk_params) 
        st = st.reshape(st.shape[0])
        pre_kp, cur_kp = ref_kp[st == 1], cur_kp[st == 1]

        return pre_kp, cur_kp
        
    def featureFiltering(self, pre_kp, cur_kp):

        pre_kp_new = np.array([[0,0]])
        cur_kp_new = np.array([[0,0]])
        of_filter = abs(pre_kp - cur_kp)
        
        for i, num in enumerate(of_filter):
            if num[0] < 30 :
                pre_kp_new = np.vstack((pre_kp_new,pre_kp[i]))
                cur_kp_new = np.vstack((cur_kp_new,cur_kp[i]))
            else:
                pass

        pre_kp_new = np.delete(pre_kp_new,0,0)
        cur_kp_new = np.delete(cur_kp_new,0,0)

        return pre_kp_new, cur_kp_new 

    def mv_Calculator(self,pre_kp_new, cur_kp_new):

        magnitude_dy = []

        for p_kp, c_kp in zip(pre_kp_new, cur_kp_new):
        
            p_kp_x ,p_kp_y = p_kp[0], p_kp[1]
            c_kp_x ,c_kp_y = c_kp[0], c_kp[1]

            magnitude_dy.append(c_kp_y-p_kp_y)
                    
        vectors = []
        vectors = cur_kp_new - pre_kp_new
        angles = [round(math.atan2(vector[0], vector[1]) * 180.0 / math.pi) for vector in vectors]          
        
        list_count_sum = [0]*360
        list_count_buf = [0]*360
        list_count = [0]*360

        list_dxdy_sum = [0]*360
        list_dxdy_buf = [0]*360
        list_dxdy = [0]*360
        
        list_dy_sum = [0]*360        
        list_dy_buf = [0]*360        
        list_dy = [0]*360
        deg = 360
        angle = np.array([])
        
        max_count = 0
        max_degree = 0
        max_dy = 0

        for i, ang in enumerate(angles):
            if int(ang) == -180:
                ang = 180
            idx = int(ang) + 179       # [0]  ~ [359]
            list_count_sum[idx] += 1             
            list_dy_sum[idx] += magnitude_dy[i]

        for i in range(deg):
            list_count_buf = list_count_sum[i % deg]
            list_count_buf = list_count_buf + list_count_sum[(i-2) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i-1) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i+2) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i+1) % deg]  
            list_count[i] = list_count_buf
            
            list_dy_buf = list_dy_sum[i % deg]
            list_dy_buf = list_dy_buf + list_dy_sum[(i-2) % deg]  
            list_dy_buf = list_dy_buf + list_dy_sum[(i-1) % deg]  
            list_dy_buf = list_dy_buf + list_dy_sum[(i+2) % deg]  
            list_dy_buf = list_dy_buf + list_dy_sum[(i+1) % deg]  
            list_dy[i] = list_dy_buf

        for idx in range(deg):
            if list_count[idx] != 0:
                list_dy[idx] /= list_count[idx]
            else:
                pass

        for i in np.arange(-179,181):
            angle = np.append(angle, i)

        max_count = max(list_count)
        max_index = list_count.index(max_count)
        max_degree = angle[max_index]
        max_dy = round(list_dy[max_index])
        
        IMU_roll, IMU_pitch = round(self.ROS.cur_imu[0],4), round(self.ROS.cur_imu[1],4)
        IMU_diff = (IMU_pitch- self.pre_IMU_pitch)/1
        self.pre_IMU_pitch = IMU_pitch

        return max_dy

    def euler2dist(self, dy):
   
        cam_temp = 0
        cam_result = []
        weight = 0.0008
        bias = 0.0001
        cam_temp = weight*dy - bias
        self.dy_sum += cam_temp
        cam_result = [0,self.dy_sum,0]
        imu_result = [0, self.ROS.cur_imu[1], 0]
        
        self.ROS.pose2ROS(cam_result,'cam')
        self.ROS.pose2ROS(imu_result,'imu')

        print("data logging!")
        # logging ori_dist, dist, GT_dist here!!!!!!!!!!!!!!!
        df = pd.DataFrame([{'1. uncorreted dist' : round(self.ROS.ori_dist,3), '2. corrected dist' : round(self.ROS.dist,3), '3. IMU dist' : round(self.ROS.dist_imu,3), '4. LiDAR GT ' : round(self.ROS.GT_dist,3)}])
        if not os.path.exists('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.12.27/logging.csv'):
            df.to_csv('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.12.27/logging.csv', index=False, mode='w')
        else:
            df.to_csv('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.12.27/logging.csv', index=False, mode='a', header=False)

        
    def main(self):
        try:
            moving_fps = 0.0
            while not rospy.is_shutdown():
                if self.ROS.get_new_IMG_msg and self.ROS.get_new_imu_msg:

                    start_t = timeit.default_timer()
                    curr_frame = cv2.cvtColor(self.ROS.cur_img['img'], cv2.COLOR_BGR2GRAY)
                    curr_kp_orig = self.featureDetection(curr_frame)

                    if self.preproc_ct != 0 :
                        prev_kp, curr_kp = self.featureTracking(prev_frame, curr_frame, prev_kp)

                        img_mask = self.ROS.cur_img['img'].copy()

                        prev_kp_new, curr_kp_new = self.featureFiltering(prev_kp, curr_kp)
                        
                        HH, inoutliers = cv2.findHomography(prev_kp_new,curr_kp_new,cv2.RANSAC,3)

                        for i,good in enumerate(inoutliers):
                            color = (0,255,0) if good else (0,0,255)
                            cv2.line(img_mask, tuple(map(int,prev_kp_new[i,:])), tuple(map(int,curr_kp_new[i,:])), color, 2, cv2.LINE_AA)
                            cv2.circle(img_mask,tuple(map(int,curr_kp_new[i,:])),3,(0,255,0),-1)

                        inoutliers = inoutliers.reshape(inoutliers.shape[0])
                        prev_inliers = prev_kp_new[inoutliers==1]
                        curr_inliers = curr_kp_new[inoutliers==1]

                        self.max_dy_ori = self.mv_Calculator(prev_inliers, curr_inliers)
                        self.euler2dist(self.max_dy_ori)

                        if self.ROS.pub_opti_res.get_num_connections() > 0:
                            self.ROS.img2ROS(img_mask)
                
                        prev_kp = curr_kp
                        
                    prev_frame = curr_frame.copy()
                    prev_kp = curr_kp_orig.copy()

                    self.img_cnt += 1
                    self.preproc_ct += 1
                    self.ROS.get_new_IMG_msg = False
                    self.ROS.get_new_imu_msg = False
                  
                    terminate_t = timeit.default_timer()
                    FPS = int(1./(terminate_t - start_t))
                    print("FPS : {}".format(FPS))
                    
        except rospy.ROSInterruptException:
            rospy.logfatal("{Opticalflow} is dead.")







