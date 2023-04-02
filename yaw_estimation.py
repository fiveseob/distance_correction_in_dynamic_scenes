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
import traceback

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray

from scipy.spatial.transform import Rotation as sci_R

import matplotlib.pyplot as plt

from geometry_msgs.msg import Transform
from std_msgs.msg import Bool

from calibration.calib import Calibration
import ROS_tools

class PoseEstimation:
    def __init__(self):

        # self.= ROS_tools.ROS(calibration)

        self.K = np.array([[1257.3423148775692, 0.00000000e+00, 691.5344282595876], 
                        [0.00000000e+00, 1122.4588251747741, 401.05061471539386], 
                        [0.0, 0.0, 1.0]])

        self.D = np.array([-0.3508815727891853, 0.31508796437915887, -0.003153745267407433, -0.001778430626275458, -0.2821173559399535])

        self.bridge = CvBridge()

        rospy.init_node('opticalflow')

        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.IMGcallback)
        rospy.Subscriber('/vectornav/IMU', Imu, self.IMUcallback, queue_size=10)

        self.pub_opti_res = rospy.Publisher('/Camera/Opticalflow', Image, queue_size=1)
        
        
        self.cur_img = {'img':None, 'header':None}
        self.get_new_IMG_msg = False
        self.get_new_imu_msg = False
        self.pose_flag = True

        self.preproc_ct = 0
        self.img_cnt = 0.0

        self.init_pose = 0
        self.pre_IMU_pitch = 0
        self.pre_IMU_yaw = 0
        self.IMU_roll = 0
        self.IMU_pitch = 0
        self.IMU_pitch_diff = 0

        self.dx_sum = 0

    def undistort(self, img):
        w,h = (img.shape[1], img.shape[0])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 0)
        result_img = cv2.undistort(img, self.K, self.D, None, newcameramtx)
        return result_img

    def IMGcallback(self, msg):
        if not self.get_new_IMG_msg:
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (1280, 806))
            # img = cv2.bilateralFilter(img, 6, 25, 25)
            self.cur_img['img'] = self.undistort(img)
            self.cur_img['header'] = msg.header
            self.get_new_IMG_msg = True

    def IMUcallback(self, msg):
        if not self.get_new_imu_msg:

            quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)

            cur_imu = list(euler)

            if self.pose_flag and cur_imu[1] != 0:
                self.init_pose = cur_imu
                self.pose_flag = False

            # self.cur_imu = [cur_imu[0]-self.init_pose[0], cur_imu[1]-self.init_pose[1], cur_imu[2]-self.init_pose[2]]
            self.cur_imu = [cur_imu[0]-self.init_pose[0], cur_imu[1]-self.init_pose[1], cur_imu[2]]

            self.IMU_roll, self.IMU_pitch, self.IMU_yaw = self.cur_imu[0], self.cur_imu[1], self.cur_imu[2]
            self.IMU_pitch_diff = (self.IMU_pitch - self.pre_IMU_pitch)
            self.IMU_yaw_diff = abs((self.IMU_yaw - self.pre_IMU_yaw))

            print("-"*35)

            # print("init_pose : {}".format(math.degrees(self.init_pose[1])))
            # print("IMU_pitch_diff : {}".format(math.degrees(round(self.IMU_pitch_diff,4))))
            # print("IMU_pitch : {}".format(math.degrees(round(self.IMU_pitch,4))))
            print("IMU_yaw_diff : {}".format(math.degrees(self.IMU_yaw_diff)))
            print("IMU_yaw : {}".format(math.degrees(self.IMU_yaw)))

            self.pre_IMU_pitch = self.IMU_pitch
            self.pre_IMU_yaw = self.IMU_yaw

            self.get_new_imu_msg = True

    
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

        magnitude_dxdy = []

        for p_kp, c_kp in zip(pre_kp_new, cur_kp_new):
        
            p_kp_x ,p_kp_y = p_kp[0], p_kp[1]
            c_kp_x ,c_kp_y = c_kp[0], c_kp[1]

            # magnitude_dy.append(c_kp_y-p_kp_y)
            # magnitude_dx.append(c_kp_x-p_kp_x)
            magnitude_dxdy.append(math.sqrt(math.pow(c_kp_x-p_kp_x,2) + math.pow(c_kp_y-p_kp_y,2)))
            
        vectors = []
        vectors = cur_kp_new - pre_kp_new
        angles = [round(math.atan2(vector[0], vector[1]) * 180.0 / math.pi) for vector in vectors]          
        
        list_count_sum = [0]*360
        list_count_buf = [0]*360
        list_count = [0]*360

        list_dxdy_sum = [0]*360
        list_dxdy_buf = [0]*360
        list_dxdy = [0]*360
        
        list_dx_sum = [0]*360        
        list_dx_buf = [0]*360        
        list_dx = [0]*360
        deg = 360
        angle = np.array([])
        
        max_count = 0
        max_degree = 0
        max_dx = 0

        for i, ang in enumerate(angles):
            if int(ang) == -180:
                ang = 180
            idx = int(ang) + 179       # [0]  ~ [359]
            list_count_sum[idx] += 1             
            list_dx_sum[idx] += magnitude_dxdy[i]

        for i in range(deg):
            list_count_buf = list_count_sum[i % deg]
            list_count_buf = list_count_buf + list_count_sum[(i-2) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i-1) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i+2) % deg]  
            list_count_buf = list_count_buf + list_count_sum[(i+1) % deg]  
            list_count[i] = list_count_buf
            
            list_dx_buf = list_dx_sum[i % deg]
            list_dx_buf = list_dx_buf + list_dx_sum[(i-2) % deg]  
            list_dx_buf = list_dx_buf + list_dx_sum[(i-1) % deg]  
            list_dx_buf = list_dx_buf + list_dx_sum[(i+2) % deg]  
            list_dx_buf = list_dx_buf + list_dx_sum[(i+1) % deg]  
            list_dx[i] = list_dx_buf

        for idx in range(deg):
            if list_count[idx] != 0:
                list_dx[idx] /= list_count[idx]
            else:
                pass

        for i in np.arange(-179,181):
            angle = np.append(angle, i)

        max_count = max(list_count)
        max_index = list_count.index(max_count)
        max_degree = angle[max_index]
        max_dx = round(list_dx[max_index])

        ### make histogram

        # plt.cla()

        # # plt.ylim(0., 650.)
        # # plt.ylim(0., 300.)
        # plt.ylim(0., 50.)

        # plt.xlim(-179, 180)
        
        # plt.suptitle('angle histogram', fontsize=16)
        # plt.xlabel('vector angle(deg)')
        # plt.ylabel('feature counting(num)')
        # # plt.scatter(angle, list_dy, c='blue')
        # # plt.bar(angle, list_count, width=5)
        # plt.bar(angle, list_dx, width=5)

        # # plt.savefig("./histogram/result{}.png".format(self.img_cnt))
        # plt.pause(0.00001)

        return max_dx, max_degree

    def euler2dist(self, dx, max_degree):
   
        cam_temp = 0
        cam_result = []
        weight = 0.0008
        bias = 0.0001
        cam_temp = weight*dx + bias
        self.dx_sum += cam_temp
        cam_result = [0,self.dx_sum,0]
        cur_euler_cam = [self.cur_imu[0], self.cur_imu[1], self.cur_imu[2]]
        # self.pose2ROS(cam_result)

        # df = pd.DataFrame([{'1.IMU_pitch(deg)': math.degrees(round(self.IMU_pitch,4)), '2.IMU_diff (deg)': math.degrees(round(self.IMU_pitch_diff,4)), '3.Max Degree(deg)': max_degree, '4.Mag dx(px)': dx}])
        df = pd.DataFrame([{'1.IMU_yaw(deg)': round(math.degrees(self.IMU_yaw),4), '2.IMU_diff (deg)': round(math.degrees(self.IMU_yaw_diff),4), '3.Max Degree(deg)': abs(max_degree), '4.Mag dx(px)': abs(dx)}])
        
        if not os.path.exists('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.11.16/logging.csv'):
            df.to_csv('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.11.16/logging.csv', index=False, mode='w')
        else:
            df.to_csv('/home/fiveseob/Lab_Project/NGV/NGV_test/logging_data/22.11.16/logging.csv', index=False, mode='a', header=False)

        print("max_degree : {}".format(max_degree))
        print("max_dx : {}".format(dx))

    def main(self):
        # try:
        moving_fps = 0.0
        while not rospy.is_shutdown():
            try:
                if self.get_new_IMG_msg and self.get_new_imu_msg:

                    max_dx, max_degree = 0, 0
                    start_fps = time.time()
                    start_t = timeit.default_timer()
                    curr_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY)
                    curr_kp_orig = self.featureDetection(curr_frame)

                    if self.preproc_ct != 0 :
                        prev_kp, curr_kp = self.featureTracking(prev_frame, curr_frame, prev_kp)

                        img_mask = self.cur_img['img'].copy()

                        prev_kp_new, curr_kp_new = self.featureFiltering(prev_kp, curr_kp)
                        
                        HH, inoutliers = cv2.findHomography(prev_kp_new,curr_kp_new,cv2.RANSAC,10)
                        for i,good in enumerate(inoutliers):
                            color = (0,255,0) if good else (0,0,255)
                            cv2.line(img_mask, tuple(map(int,prev_kp_new[i,:])), tuple(map(int,curr_kp_new[i,:])), color, 2, cv2.LINE_AA)
                            cv2.circle(img_mask,tuple(map(int,curr_kp_new[i,:])),3,(0,255,0),-1)

                        inoutliers = inoutliers.reshape(inoutliers.shape[0])
                        prev_inliers = prev_kp_new[inoutliers==1]
                        curr_inliers = curr_kp_new[inoutliers==1]

                        max_dx, max_degree = self.mv_Calculator(prev_inliers, curr_inliers)
                        self.euler2dist(max_dx, max_degree)

                        if self.pub_opti_res.get_num_connections() > 0:
                            msg = None
                            msg = self.bridge.cv2_to_imgmsg(img_mask,"bgr8")
                            msg.header = self.cur_img['header']
                            self.pub_opti_res.publish(msg)
                
                        prev_kp = curr_kp
                        
                    prev_frame = curr_frame.copy()
                    prev_kp = curr_kp_orig.copy()

                    self.img_cnt += 1
                    self.preproc_ct += 1
                    self.get_new_IMG_msg = False
                    self.get_new_imu_msg = False
                    
                    end_fps = time.time() - start_fps
                    if self.img_cnt != 0:
                        moving_fps = (self.img_cnt / float(self.img_cnt + 1) * moving_fps) + (1. / float(self.img_cnt + 1) * end_fps)
                        # print("FPS_averge : {%0.2f}" %(1./moving_fps))
                    terminate_t = timeit.default_timer()
                    FPS = int(1./(terminate_t - start_t))
                    # print("FPS : {}".format(FPS))
            except Exception:
                prev_frame = curr_frame.copy()
                prev_kp = curr_kp_orig.copy()
                traceback.print_exc() 
                # print("error")
                # rospy.logwarn("{Opticalflow} is dead.")



if __name__ == "__main__":
    optical = PoseEstimation()
    optical.main()
    rospy.spin()


