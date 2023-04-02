import sys
import os
import time
import argparse
import numpy as np
import cv2
import copy
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *

from API.tracker import Tracker
from API.drawer import Drawer
from calibration.calib import Calibration
from dist import Distance

import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Transform
from std_msgs.msg import Float32MultiArray


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def detect(engine, context, buffers, image_src, input_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = input_size

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
  
    inputs, outputs, bindings, stream = buffers
   
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    trt_output = trt_outputs[0].reshape(1, -1, 4 + num_classes)

    tb = time.time()

    boxes = post_processing(img_in, 0.7, 0.6, trt_output)

    return boxes

def LiDARcallback(msg):
    global get_new_LiDAR_msg
    global GT_dist

    for marker in msg.markers:
        lidar_x = -marker.pose.position.x
        lidar_y = marker.pose.position.y
        lidar_z = marker.pose.position.z
    
    if len(msg.markers) > 0:
        GT_dist = math.sqrt(lidar_x**2+lidar_z**2)

    get_new_LiDAR_msg = True

def Posecallback(msg):
    global cur_pose
    cur_pose = [msg.rotation.x, msg.rotation.y, msg.rotation.z]

def Right_IMGcallback(msg):
    global cur_img
    global calib
    global get_new_IMG_msg
    if not get_new_IMG_msg:
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 806))

        cur_img['img'] = calib.undistort(img, 'right')
        cur_img['header'] = msg.header
        get_new_IMG_msg = True
      
def get_3d_box_array_msg(pts_3d):
    pt_3d_array_msg = Float32MultiArray()
    pt_3d_info = []
    for idx, pt in enumerate(pts_3d):
        Real_X = pt[0]
        Real_Y = pt[1]
        pt_3d_info.append([Real_X, Real_Y])

    pt_3d_array_msg.data = sum(pt_3d_info, [])

    return pt_3d_array_msg

def get_dist_array_msg(dist_arr):
        dist_array_msg = Float32MultiArray()
        
        for dist in dist_arr:
            dist_array_msg.data.append(dist[0])

        return dist_array_msg

if __name__ == '__main__':
    TRT_LOGGER = trt.Logger()
    ### config ###
    engine_path = "./weight/0927"#./weight/Yolov4_epoch231"
    input_size = (320, 320)
    image_shape=(1280, 806)
    num_classes = 80
    namesfile = 'data/coco.names'
    interval = 1

    camera_path = ['calibration/f_camera_1280.txt', 'calibration/l_camera_1280.txt', 'calibration/r_camera_1280.txt']
    imu_camera_path = 'calibration/camera_imu.txt'
    LiDAR_camera_path = 'calibration/f_camera_lidar_1280.txt'

    cur_img = {'img':None, 'header':None}
    dist_arr = []
    GT_dist = None
    cur_pose = None
    get_new_IMG_msg = False
    get_new_LiDAR_msg = False

    tracker = Tracker(image_shape, min_hits=1, num_classes=num_classes, interval=interval)  # Initialize tracker
    drawer = Drawer(namesfile)
    calib = Calibration(os.path.join(camera_path), os.path.join(imu_camera_path), os.path.join(LiDAR_camera_path))
    est_dist = Distance(calib)
    bridge = CvBridge()

    rospy.init_node('detection_right')
    rospy.Subscriber('/cam1/compressed', CompressedImage, Right_IMGcallback)
    rospy.Subscriber('/lidar/postpoint', MarkerArray, LiDARcallback, queue_size=10)
    rospy.Subscriber('/Camera/Transform', Transform, Posecallback, queue_size=10)

    pub_img = rospy.Publisher('/detection_right/result_right', Image, queue_size=1)
    pub_dist = rospy.Publisher("/detection_right/dist", Float32MultiArray, queue_size=1)
    pub_ori_dist = rospy.Publisher("/detection_right/ori_dist", Float32MultiArray, queue_size=1)
    
    pub_2d_bbox = rospy.Publisher('/detection_right/bbox_2d', Float32MultiArray, queue_size=1)
    pub_3d_bbox = rospy.Publisher('/detection_right/bbox_3d', Float32MultiArray, queue_size=1)


    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine)

        try:
            new_dist_flag = False
            frame_ind = 0
            moving_tra, moving_det = 0., 0.
            ground_plane = np.array(calib.r_src_pt, np.int32)
            
            while not rospy.is_shutdown():
                if get_new_IMG_msg:
                    start = time.time()
                    dets_arr, labels_arr, is_dect = None, None, None
                    if np.mod(frame_ind, interval) == 0:
                        img = cv2.resize(cur_img['img'], input_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        orig_im = copy.copy(cur_img['img'])

                        bbox = detect(engine, context, buffers, img, input_size, num_classes)[0]
                        
                        if len(bbox) > 0: 
                            bbox = np.vstack(bbox)
                            output = copy.copy(bbox)
                            
                            output[:,0] = (bbox[:,0] - bbox[:,2] / 2.0) * image_shape[0]
                            output[:,1] = (bbox[:,1] - bbox[:,3] / 2.0) * image_shape[1]
                            output[:,2] = (bbox[:,0] + bbox[:,2] / 2.0) * image_shape[0]
                            output[:,3] = (bbox[:,1] + bbox[:,3] / 2.0) * image_shape[1]

                            dets_arr, labels_arr = output[:,0:4], output[:,-1].astype(int)
                            ori_dist_arr, dist_arr, pts_3d, ground_plane, new_dist_flag = est_dist.getDistance(dets_arr, cur_pose, 'right')

                            pub_dist.publish(get_dist_array_msg(dist_arr))
                            pub_ori_dist.publish(get_dist_array_msg(ori_dist_arr))

                            ## Compare for LiDAR
                            pub_3d_bbox.publish(get_3d_box_array_msg(pts_3d))

                        else:
                            dets_arr, labels_arr = np.array([]), np.array([])
                            
                        is_dect = True

                    elif np.mod(frame_ind, interval) != 0:
                        dets_arr, labels_arr = np.array([]), np.array([])
                        is_dect = False

                    pt_det = (time.time() - start)

                    tracker_arr = tracker.update(dets_arr, labels_arr, is_dect=is_dect)
                    tracker_arr, labels_arr = tracker_arr[:,0:4], tracker_arr[:,-1].astype(int)


                    if len(est_dist.center_pts) > 0:
                        for pt in est_dist.center_pts:
                            orig_im = cv2.line(orig_im, (int(pt[0]), int(pt[1])),(int(pt[0]), int(pt[1])), (255,255,255), 10)

                    if frame_ind != 0:
                        moving_det = (frame_ind / float(frame_ind + 1) * moving_det) + (1. / float(frame_ind + 1) * pt_det)

                    show_frame = drawer.draw(orig_im, tracker_arr, labels_arr, dist_arr, GT_dist, (1. / (moving_det + 1e-8)), is_tracker=False)

                    if new_dist_flag:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,0,255), thickness=2)
                    else:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,255,0), thickness=2)
                    
                    if pub_img.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = bridge.cv2_to_imgmsg(show_frame, "bgr8")
                            msg.header = cur_img['header']
                        except CvBridgeError as e:
                            print(e)
                        pub_img.publish(msg)

                    frame_ind += 1
                    get_new_IMG_msg = False
                    new_dist_flag = False

        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")
