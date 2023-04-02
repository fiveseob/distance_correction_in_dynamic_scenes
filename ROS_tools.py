import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform
from std_msgs.msg import Bool

class ROS:
    def __init__(self, calibration):
        self.calib = calibration

        self.bridge = CvBridge()
        rospy.init_node('main')

        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.IMGcallback)
        rospy.Subscriber('/vectornav/IMU', Imu, self.IMUcallback, queue_size=10)

        rospy.Subscriber('/Camera/Transform', Transform, self.Posecallback, queue_size=10)
        
        rospy.Subscriber('/detection/dist_imu', Float32MultiArray, self.ImuDistcallback, queue_size=10)
        rospy.Subscriber('/detection/dist', Float32MultiArray, self.Distcallback, queue_size=10)
        rospy.Subscriber('/detection/ori_dist', Float32MultiArray, self.Ori_Distcallback, queue_size=10)
        rospy.Subscriber('/lidar/postpoint', MarkerArray, self.LiDARcallback, queue_size=10)
        self.pub_rawimg = rospy.Publisher('/compressed_image', CompressedImage, queue_size=1)
        self.pub_imu = rospy.Publisher('/detection/imu', Imu, queue_size=1)
        self.pub_pose = rospy.Publisher('/Camera/Transform', Transform, queue_size=1)
        self.pub_pose_imu = rospy.Publisher('/Imu/Transform', Transform, queue_size=1)
        
        self.pub_opti_res = rospy.Publisher('/Camera/Opticalflow', Image, queue_size=1)

        self.cur_img = {'img':None, 'header':None}
        self.left_cur_img = {'img':None, 'header':None}
        self.right_cur_img = {'img':None, 'header':None}

        self.cur_imu = None
        self.cur_pose  = None
        self.cur_LiDAR = None

        self.get_new_IMG_msg = False
        self.get_new_Right_IMG_msg = False
        self.get_new_Left_IMG_msg = False
        self.get_new_pose_msg = False
        self.get_new_LiDAR_msg = False
        self.get_new_imu_msg = False
        self.get_DT_msg = False
        
        self.bboxes = []
        self.pose_flag = True

        self.dist = 0
        self.dist_imu = 0
        self.ori_dist = 0
        self.GT_dist = 0
    
    def LiDARcallback(self, msg):
        for marker in msg.markers:
            lidar_x = -marker.pose.position.x
            lidar_y = marker.pose.position.y
            lidar_z = marker.pose.position.z

        if len(msg.markers) > 0:
            diag = math.sqrt(lidar_x**2+lidar_y**2)
            self.GT_dist = math.sqrt(diag**2 + lidar_z**2)

        self.get_new_LiDAR_msg = True

    def ImuDistcallback(self, msg):
        self.dist_imu = msg.data[0]

    def Distcallback(self, msg):
        self.dist = msg.data[0]

    def Ori_Distcallback(self, msg):
        if len(msg.data) != 0:
            self.ori_dist = msg.data[0]

    def IMGcallback(self, msg):
        if not self.get_new_IMG_msg:

            self.pub_rawimg.publish(msg)

            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (1280, 806))
            
            self.cur_img['img'] = self.calib.undistort(img, 'front')
            self.cur_img['header'] = msg.header
            self.get_new_IMG_msg = True

    def Left_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        self.left_cur_img['img'] = self.calib.undistort(img, 'left')
        self.left_cur_img['header'] = msg.header
        self.get_new_Left_IMG_msg = True

    def Right_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        self.right_cur_img['img'] = self.calib.undistort(img, 'right')
        self.right_cur_img['header'] = msg.header
        self.get_new_Right_IMG_msg = True


    def IMUcallback(self, msg):
        if not self.get_new_imu_msg:

            ## quat to euler
            self.pub_imu.publish(msg)

            x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
           
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.copysign(3.141592 / 2, sinp)
            else:
                pitch = math.asin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            cur_imu = [0, pitch, 0]
            
            if self.pose_flag and cur_imu[1] != 0:
                self.init_pose = cur_imu
                self.pose_flag = False

            self.cur_imu = [0, pitch-self.init_pose[1], 0]
            self.get_new_imu_msg = True

    
    def Posecallback(self, msg):
        self.cur_pose  = [msg.rotation.x, msg.rotation.y, msg.rotation.z]
        ## radian
        self.Euler = [self.cur_pose[0], self.cur_pose[1], self.cur_pose[2]] ## roll pitch yaw

        self.get_new_pose_msg = True

    ### Publisher
    def pose2ROS(self, cur_euler, task):
        pub_pose_msg = Transform()
      
        pub_pose_msg.rotation.x = cur_euler[0]
        pub_pose_msg.rotation.y = cur_euler[1]
        pub_pose_msg.rotation.z = cur_euler[2]
        pub_pose_msg.rotation.w = 0

        if task == 'cam':
            self.pub_pose.publish(pub_pose_msg)
        elif task == 'imu':
            print("imu publish ! ")
            self.pub_pose_imu.publish(pub_pose_msg)    

    def img2ROS(self, img):
        msg = None
        try:
            msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            msg.header = self.cur_img['header']
        except CvBridgeError as e:
            print(e)
        self.pub_opti_res.publish(msg)
       