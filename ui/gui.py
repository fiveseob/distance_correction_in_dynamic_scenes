import re
import numpy as np
import cv2
import os

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge

from queue import Queue
import inspect
import copy

import math
import tf

g_imgBig = np.zeros((3,3,3),dtype=np.uint8)
bridge = CvBridge()

img_qsize = 2
g_qsize = 10
queue_img_front = Queue()
queue_img_left = Queue()
queue_img_right = Queue()

g_img_front_last = np.zeros((238,423,3),dtype=np.uint8)
g_img_left_last = np.zeros((238,423,3),dtype=np.uint8)
g_img_right_last = np.zeros((238,423,3),dtype=np.uint8)

g_imu_init_pose = None
g_imu_pose_flag = True
queue_degrees_imu = Queue()
g_degree_imu = [0,0,0]
queue_degrees_cam = Queue()
g_degree_cam = [0,0,0]

queue_left_distance_gt = Queue()
queue_left_distance_before = Queue()
queue_left_distance_after = Queue()
queue_front_distance_gt = Queue()
queue_front_distance_before = Queue()
queue_front_distance_after = Queue()
queue_right_distance_gt = Queue()
queue_right_distance_before = Queue()
queue_right_distance_after = Queue()

g_left_distance_before = -1.7
g_left_distance_gt = -1.7
g_left_distance_after = -1.7
g_front_distance_gt = -1.7
g_front_distance_before = -1.7
g_front_distance_after = -1.7
g_right_distance_gt = -1.7
g_right_distance_before = -1.7
g_right_distance_after = -1.7

g_count_front_gt = 0
g_count_front_before = 0
g_count_front_after = 0
g_count_left_gt = 0
g_count_left_before = 0
g_count_left_after = 0
g_count_right_gt = 0
g_count_right_before = 0
g_count_right_after = 0
g_count_limit = 20

g_list_imu_pitch = []
g_list_cam_pitch = []

recording_toggle = False
recording_init_toggle = False
toggle = True

dir_path = os.path.dirname(os.path.realpath(__file__))
filename='result'


def overlay_cam_front():
    global g_img_front_last
    myX, myY, myW, myH = 750, 255, 423, 238
    if not queue_img_front.empty():
        img_front_src = queue_img_front.get()
        img_front_small = cv2.resize(img_front_src, (myW, myH))
        g_img_front_last = copy.copy(img_front_small)
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_front_small,None)
    else:
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(g_img_front_last,None)


def overlay_cam_left():
    global g_img_left_last
    myX, myY, myW, myH = 247, 255, 423, 238
    if not queue_img_left.empty():
        img_left_src = queue_img_left.get()
        img_left_small = cv2.resize(img_left_src, (myW, myH))
        g_img_left_last = copy.copy(img_left_small)
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_left_small,None)
    else:
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(g_img_left_last,None)

def overlay_cam_right():
    global g_img_right_last
    myX, myY, myW, myH = 1267, 255, 423, 238
    if not queue_img_right.empty():
        img_right_src = queue_img_right.get()
        img_right_small = cv2.resize(img_right_src, (myW, myH))
        g_img_right_last = copy.copy(img_right_small)
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_right_small,None)
    else:
        g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(g_img_right_last,None)

def cal_error(gt, distance):
    try:
        error = 100*(abs(distance)-abs(gt))/abs(gt)
    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))
    return round(error, 2)

def overlay_text_front():
    global g_front_distance_gt, g_front_distance_before, g_front_distance_after, \
    g_count_front_gt, g_count_front_before, g_count_front_after, g_count_limit

    
    gt_X, gt_Y, before_X, before_Y, after_X, after_Y = 1040, 584, 940, 642, 940, 708
    error_before_X, error_before_Y, error_after_X, error_after_Y = 1085, 642, 1085, 708


    if queue_front_distance_gt.qsize() > g_qsize-2:
        g_front_distance_gt = round(np.mean(np.array(queue_front_distance_gt.queue)),1)
        queue_front_distance_gt.get()
        g_count_front_gt = 0
    else:
        if g_count_front_gt < g_count_limit:
            g_count_front_gt += 1

    if queue_front_distance_before.qsize() > g_qsize-2:
        g_front_distance_before = round(np.mean(np.array(queue_front_distance_before.queue)),1)
        queue_front_distance_before.get()
        g_count_front_before = 0
    else:
        if g_count_front_before < g_count_limit:
            g_count_front_before += 1

    if queue_front_distance_after.qsize() > g_qsize-2:
        g_front_distance_after = round(np.mean(np.array(queue_front_distance_after.queue)),1)
        queue_front_distance_after.get()
        g_count_front_after = 0
    else:
        if g_count_front_after < g_count_limit:
            g_count_front_after += 1

    gt_text = '-' if g_front_distance_gt == -1.7 else str(g_front_distance_gt)
    # gt_color = (0,0,255) if g_count_front_gt == g_count_limit else (156, 138, 98)
    gt_color = (156, 138, 98) if g_count_front_gt == g_count_limit else (156, 138, 98)
    before_text = '-' if g_front_distance_before == -1.7 else str(g_front_distance_before)
    # before_color = (0,0,255) if g_count_front_before == g_count_limit else (156, 138, 98)
    before_color = (156, 138, 98) if g_count_front_before == g_count_limit else (156, 138, 98)
    after_text = '-' if g_front_distance_after == -1.7 else str(g_front_distance_after)
    # after_color = (0,0,255) if g_count_front_after == g_count_limit else (156, 138, 98)
    after_color = (156, 138, 98) if g_count_front_after == g_count_limit else (156, 138, 98)

    cv2.putText(g_imgBig,gt_text,(gt_X,gt_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,gt_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,before_text,(before_X,before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,before_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,after_text,(after_X,after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,after_color,2,lineType=cv2.LINE_AA)

    error_before = cal_error(g_front_distance_gt, g_front_distance_before)    
    error_after = cal_error(g_front_distance_gt, g_front_distance_after)

    if error_before <= 5 and error_before >= -5:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)    
    else:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)    
    if error_after <= 5 and error_after >= -5:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)

def overlay_text_left():
    global g_left_distance_gt, g_left_distance_before, g_left_distance_after, \
    g_count_left_gt, g_count_left_before, g_count_left_after, g_count_limit
    
    gt_X, gt_Y, before_X, before_Y, after_X, after_Y = 540, 584, 440, 642, 440, 708
    error_before_X, error_before_Y, error_after_X, error_after_Y = 585, 642, 585, 708

    if queue_left_distance_gt.qsize() > g_qsize-2:
        g_left_distance_gt = round(np.mean(np.array(queue_left_distance_gt.queue)),1)
        queue_left_distance_gt.get()
        g_count_left_gt = 0
    else:
        if g_count_left_gt < g_count_limit:
            g_count_left_gt += 1

    if queue_left_distance_before.qsize() > g_qsize-2:
        g_left_distance_before = round(np.mean(np.array(queue_left_distance_before.queue)),1)
        queue_left_distance_before.get()
        g_count_left_before = 0
    else:
        if g_count_left_before < g_count_limit:
            g_count_left_before += 1

    if queue_left_distance_after.qsize() > g_qsize-2:
        g_left_distance_after = round(np.mean(np.array(queue_left_distance_after.queue)),1)
        queue_left_distance_after.get()
        g_count_left_after = 0
    else:
        if g_count_left_after < g_count_limit:
            g_count_left_after += 1

    gt_text = '-' if g_left_distance_gt == -1.7 else str(g_left_distance_gt)
    gt_color = (0,0,255) if g_count_left_gt == g_count_limit else (156, 138, 98)
    before_text = '-' if g_left_distance_before == -1.7 else str(g_left_distance_before)
    before_color = (0,0,255) if g_count_left_before == g_count_limit else (156, 138, 98)
    after_text = '-' if g_left_distance_after == -1.7 else str(g_left_distance_after)
    after_color = (0,0,255) if g_count_left_after == g_count_limit else (156, 138, 98)

    cv2.putText(g_imgBig,gt_text,(gt_X,gt_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,gt_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,before_text,(before_X,before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,before_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,after_text,(after_X,after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,after_color,2,lineType=cv2.LINE_AA)

    
    error_before = cal_error(g_left_distance_gt, g_left_distance_before)    
    error_after = cal_error(g_left_distance_gt, g_left_distance_after)

    if error_before <= 5 and error_before >= -5:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)    
    else:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)    
    if error_after <= 5 and error_after >= -5:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)
            

def overlay_text_right():
    global g_right_distance_gt, g_right_distance_before, g_right_distance_after, \
    g_count_right_gt, g_count_right_before, g_count_right_after, g_count_limit
    
    gt_X, gt_Y, before_X, before_Y, after_X, after_Y = 1560, 584, 1460, 642, 1460, 708
    error_before_X, error_before_Y, error_after_X, error_after_Y = 1605, 642, 1605, 708

    if queue_right_distance_gt.qsize() > g_qsize-2:
        g_right_distance_gt = round(np.mean(np.array(queue_right_distance_gt.queue)),1)
        queue_right_distance_gt.get()
        g_count_right_gt = 0
    else:
        if g_count_right_gt < g_count_limit:
            g_count_right_gt += 1

    if queue_right_distance_before.qsize() > g_qsize-2:
        g_right_distance_before = round(np.mean(np.array(queue_right_distance_before.queue)),1)
        queue_right_distance_before.get()
        g_count_right_before = 0
    else:
        if g_count_right_before < g_count_limit:
            g_count_right_before += 1

    if queue_right_distance_after.qsize() > g_qsize-2:
        g_right_distance_after = round(np.mean(np.array(queue_right_distance_after.queue)),1)
        queue_right_distance_after.get()
        g_count_right_after = 0
    else:
        if g_count_right_after < g_count_limit:
            g_count_right_after += 1

    gt_text = '-' if g_right_distance_gt == -1.7 else str(g_right_distance_gt)
    gt_color = (0,0,255) if g_count_right_gt == g_count_limit else (156, 138, 98)
    before_text = '-' if g_right_distance_before == -1.7 else str(g_right_distance_before)
    before_color = (0,0,255) if g_count_right_before == g_count_limit else (156, 138, 98)
    after_text = '-' if g_right_distance_after == -1.7 else str(g_right_distance_after)
    after_color = (0,0,255) if g_count_right_after == g_count_limit else (156, 138, 98)

    cv2.putText(g_imgBig,gt_text,(gt_X,gt_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,gt_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,before_text,(before_X,before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,before_color,2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,after_text,(after_X,after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,after_color,2,lineType=cv2.LINE_AA)

    
    error_before = cal_error(g_right_distance_gt, g_right_distance_before)    
    error_after = cal_error(g_right_distance_gt, g_right_distance_after)

    if error_before <= 5 and error_before >= -5:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)    
    else:
        cv2.putText(g_imgBig,str(error_before),(error_before_X, error_before_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)    
    if error_after <= 5 and error_after >= -5:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(248, 210, 43),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(g_imgBig,str(error_after),(error_after_X, error_after_Y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(155, 85, 249),2,lineType=cv2.LINE_AA)


def callback_cam_left(msg):
    global img_qsize    
    if queue_img_left.qsize() < img_qsize:
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = bridge.imgmsg_to_cv2(msg, "bgr8")       
        queue_img_left.put(img)

def callback_cam_front(msg):
    global img_qsize
    if queue_img_front.qsize() < img_qsize:
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    	
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        queue_img_front.put(img)

def callback_cam_right(msg):
    global img_qsize
    if queue_img_right.qsize() < img_qsize:
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = bridge.imgmsg_to_cv2(msg, "bgr8")        
        queue_img_right.put(img)

def front_distance_before_callback(msg):
    try:
        if queue_front_distance_before.qsize() < g_qsize:
            distance = copy.deepcopy(msg.data[0])
            queue_front_distance_before.put(distance)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def front_distance_after_callback(msg):
    try:
        if queue_front_distance_after.qsize() < g_qsize:
            distance = copy.deepcopy(msg.data[0])
            queue_front_distance_after.put(distance)
            toggle = True

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def left_distance_before_callback(msg):
    try:
        if queue_left_distance_before.qsize() < g_qsize:
            distance = copy.deepcopy(round(msg.data[0],2))
            queue_left_distance_before.put(distance)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def left_distance_after_callback(msg):
    try:
        if queue_left_distance_after.qsize() < g_qsize:
            distance = copy.deepcopy(round(msg.data[0],2))
            queue_left_distance_after.put(distance)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def right_distance_before_callback(msg):
    try:
        if queue_right_distance_before.qsize() < g_qsize:
            distance = copy.deepcopy(round(msg.data[0],2))
            queue_right_distance_before.put(distance)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def right_distance_after_callback(msg):
    try:
        if queue_right_distance_after.qsize() < g_qsize:
            distance = copy.deepcopy(round(msg.data[0],2))
            queue_right_distance_after.put(distance)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def LiDARcallback(msg):
    try:
        if queue_front_distance_gt.qsize() < g_qsize:
            for marker in msg.markers:
                lidar_x = -marker.pose.position.x
                lidar_y = marker.pose.position.y
                lidar_z = marker.pose.position.z
            
            if len(msg.markers) > 0:
                GT_dist = math.sqrt(lidar_x**2+lidar_y**2)
                queue_front_distance_gt.put(GT_dist)
                queue_left_distance_gt.put(GT_dist)
                queue_right_distance_gt.put(GT_dist)

    except Exception as e:
        print(str(e),'line %d' % inspect.getlineno(inspect.currentframe()))

def overlay_pose_plot():
    global g_list_imu_pitch, g_list_cam_pitch, g_degree_imu, g_degree_cam
    
    myX, myY, myW, myH = 750, 780, 941, 211
    if not queue_degrees_imu.empty():
        g_degree_imu = queue_degrees_imu.get()
    if not queue_degrees_cam.empty():
        g_degree_cam = queue_degrees_cam.get()

    imu_pitch = g_degree_imu[1]
    cam_pitch = g_degree_cam[1]

    g_list_imu_pitch.append(imu_pitch)
    g_list_cam_pitch.append(cam_pitch)
    if len(g_list_imu_pitch) > 50:
        del(g_list_imu_pitch[0])
    if len(g_list_cam_pitch) > 50:
        del(g_list_cam_pitch[0])

    resolution_scale_factor = 20
    plot_x, plot_y = 50*resolution_scale_factor, 10*resolution_scale_factor+1
    img_plot = np.zeros((plot_y,plot_x,3),dtype=np.uint8)
    img_plot[:,:,0] = 156
    img_plot[:,:,1] = 138
    img_plot[:,:,2] = 98
    for x1,cam in enumerate(g_list_cam_pitch):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_cam_pitch[x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_cam_pitch[x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (248, 210, 43),thickness=2,lineType=cv2.LINE_AA)

    for x1,imu in enumerate(g_list_imu_pitch):
        if x1 == 0: continue
        x2 = x1-1
        y2 = -1*(g_list_imu_pitch[x2]*resolution_scale_factor) + plot_y/2
        y1 = -1*(g_list_imu_pitch[x1]*resolution_scale_factor) + plot_y/2
        cv2.line(img_plot, \
            (x2*resolution_scale_factor,max(min(int(y2),plot_y-1),0)), \
            (x1*resolution_scale_factor,max(min(int(y1),plot_y-1),0)), \
                (62, 52, 28),thickness=2,lineType=cv2.LINE_AA)

    img_plot_small = cv2.resize(img_plot, (myW, myH))
    g_imgBig[myY:myY+myH, myX:myX+myW] = cv2.copyTo(img_plot_small,None)
    cv2.putText(g_imgBig,'   5',(750,800),cv2.FONT_HERSHEY_SIMPLEX,0.4,(62, 52, 28),1,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,' 2.5',(750, 845),cv2.FONT_HERSHEY_SIMPLEX,0.4,(62, 52, 28),1,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'   0',(750, 890),cv2.FONT_HERSHEY_SIMPLEX,0.4,(62, 52, 28),1,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'-2.5',(750, 935),cv2.FONT_HERSHEY_SIMPLEX,0.4,(62, 52, 28),1,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'  -5',(750, 980),cv2.FONT_HERSHEY_SIMPLEX,0.4,(62, 52, 28),1,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'deg',(700, 890),cv2.FONT_HERSHEY_SIMPLEX,0.6,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'-5 sec',(750, 1005),cv2.FONT_HERSHEY_SIMPLEX,0.4,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'0 sec',(1650, 1005),cv2.FONT_HERSHEY_SIMPLEX,0.4,(156, 138, 98),2,lineType=cv2.LINE_AA)
    cv2.putText(g_imgBig,'recording : ', (1700, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(156, 138, 98),2,lineType=cv2.LINE_AA)
    if recording_toggle:
        cv2.putText(g_imgBig, 'on', (1850, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 200),2,lineType=cv2.LINE_AA)
    else:
        cv2.putText(g_imgBig, 'off', (1850, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(156, 138, 98),2,lineType=cv2.LINE_AA)   

def callback_imu(msg):
    global g_imu_pose_flag, g_imu_init_pose
    quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    cur_imu = [euler[0], euler[1], 0]

    if g_imu_pose_flag and cur_imu[1] != 0:
        g_imu_init_pose = cur_imu
        g_imu_pose_flag = False

    cur_imu = [euler[0]-g_imu_init_pose[0], euler[1]-g_imu_init_pose[1], 0]

    ## radian to degree
    roll = math.degrees(cur_imu[0])
    pitch = math.degrees(cur_imu[1])
    yaw = math.degrees(cur_imu[2])
    if queue_degrees_imu.qsize() < 2:
        queue_degrees_imu.put([roll, pitch, yaw])

def callback_img_pose(msg):
    roll = -math.degrees(round(msg.rotation.x,2))
    pitch = math.degrees(round(msg.rotation.y,2))
    yaw = -math.degrees(round(msg.rotation.z,2))
    if queue_degrees_cam.qsize() < 2:
        queue_degrees_cam.put([roll, pitch, yaw])

def output_path(mode):
    global filename

    if mode == 'log':
        file_ext = '.csv'
    elif mode == 'video':
        file_ext = '.mp4'
        # file_ext = '.avi'

    if not os.path.exists('{}'.format(dir_path) + '/results'):
       os.mkdirs('{}'.format(dir_path) + '/results')

    output_path='{}'.format(dir_path) + '/results/%s%s' %(filename,file_ext)
    uniq=1
    while os.path.exists(output_path):
        output_path = '{}'.format(dir_path) + '/results/%s_%d%s' % (filename,uniq,file_ext) 
        uniq+=1
    return output_path

def logging():
    global f, g_degree_imu, g_left_distance_gt, g_left_distance_before, g_left_distance_after, g_front_distance_gt, g_front_distance_before, g_front_distance_after, g_right_distance_gt, g_right_distance_before, g_right_distance_after
    f.write(
    str(g_degree_imu[1]) + ',' +
    str(g_left_distance_gt) + ',' + str(g_left_distance_before) + ',' + str(g_left_distance_after) + ',' +
    str(g_front_distance_gt) + ',' + str(g_front_distance_before) + ',' + str(g_front_distance_after) + ',' +
    str(g_right_distance_gt) + ',' + str(g_right_distance_before) + ',' + str(g_right_distance_after) + ',' +
    '\n')
    toggle = False

def video_recording():
    width = g_imgBig.shape[1]
    height = g_imgBig.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'divx')
    result_video_path = output_path('video')
    return cv2.VideoWriter(result_video_path, fourcc, 5.0, (int(width), int(height)))

if __name__=='__main__':
    imgBG = cv2.imread('resources/bg.png')

    rospy.init_node('GUI', anonymous=True)
    # rospy.Subscriber('/cam0/compressed', CompressedImage, callback_cam_left)
    # rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, callback_cam_front) 
    # rospy.Subscriber('/cam1/compressed', CompressedImage, callback_cam_right)

    rospy.Subscriber('/detection/result_left', Image, callback_cam_left) 
    # rospy.Subscriber('/detection/result2', Image, callback_cam_left)     
    rospy.Subscriber('/detection/result_front', Image, callback_cam_front)
    # rospy.Subscriber('/detection/result_front', Image, callback_cam_front) 	
    rospy.Subscriber('/detection/result_right', Image, callback_cam_right)  
    # rospy.Subscriber('/detection/result3', Image, callback_cam_right)

    rospy.Subscriber('/vectornav/IMU', Imu, callback_imu, queue_size=1)
    rospy.Subscriber('/Camera/Transform', Transform, callback_img_pose, queue_size=10)
    rospy.Subscriber('/detection_front/dist', Float32MultiArray, front_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_front/ori_dist', Float32MultiArray, front_distance_before_callback, queue_size=10)
    # rospy.Subscriber('/detection/dist', Float32MultiArray, front_distance_after_callback, queue_size=10)
    # rospy.Subscriber('/detection/ori_dist', Float32MultiArray, front_distance_before_callback, queue_size=10)
    rospy.Subscriber('/lidar/postpoint', MarkerArray, LiDARcallback, queue_size=10)
    rospy.Subscriber('/detection_left/dist', Float32MultiArray, left_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_left/ori_dist', Float32MultiArray, left_distance_before_callback, queue_size=10)
    rospy.Subscriber('/detection_right/dist', Float32MultiArray, right_distance_after_callback, queue_size=10)
    rospy.Subscriber('/detection_right/ori_dist', Float32MultiArray, right_distance_before_callback, queue_size=10)
  
    while not rospy.is_shutdown():
        if recording_toggle and recording_init_toggle:
            result_log_path= output_path('log')
            f = open(result_log_path, 'w')
            f.write('Pitch(degree),'
                + 'Left GT(m),Left Uncorrected Distance(m),Left Corrected Distance(m),'
                + 'Front GT(m),Front Uncorrected Distance(m),Front Corrected Distance(m),'
                + 'Right GT(m),Right Uncorrected Distance(m),Right Corrected Distance(m)'
                + '\n')
            video = video_recording()
            recording_init_toggle = False
        g_imgBig = imgBG.copy()

        overlay_cam_front()
        overlay_cam_left()
        overlay_cam_right()

        overlay_pose_plot()

        overlay_text_front()
        overlay_text_left()
        overlay_text_right()

        if recording_toggle:
            video.write(g_imgBig)
            if toggle:
                logging()

        cv2.namedWindow("pyngv", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("pyngv",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("pyngv", g_imgBig)
        try:
            key = cv2.waitKey(30)
            if key == 27:
                if recording_toggle:
                    f.close()
                    video.release()
                exit()
            if key == 114:
                if recording_toggle:
                    recording_toggle = False
                    recording_init_toggle = False
                    f.close()
                    video.release()
                else:
                    recording_toggle = True
                    recording_init_toggle = True
        except:
            pass
