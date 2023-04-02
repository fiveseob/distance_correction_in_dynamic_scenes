import cv2
import numpy as np

class Calibration:
    def __init__(self, camera_path, imu_cam_path, LiDAR_cam_path):
        if not None in [camera_path, imu_cam_path]:
            
            # camera parameters
            cam_param = []
            left_cam_param = []
            right_cam_param = []
            left_cam_rt = []
            right_cam_rt = []

            with open(camera_path[0], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        cam_param.append(float(val))

            with open(camera_path[1], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        left_cam_param.append(float(val))

            with open(camera_path[2], 'r') as f:
                for i in f.readlines():
                    for val in i.split(','):
                        right_cam_param.append(float(val))

            '''Main(Front) Camera Calibration'''
            self.camera_matrix = np.array([[cam_param[0], cam_param[1], cam_param[2]], 
                                        [cam_param[3], cam_param[4], cam_param[5]], 
                                        [cam_param[6], cam_param[7], cam_param[8]]])
            self.dist_coeffs = np.array([[cam_param[9]], [cam_param[10]], [cam_param[11]], [cam_param[12]], [cam_param[13]]])


            '''Multi-Camera parameters'''
            self.Left_camera_matrix = np.array([[left_cam_param[0], left_cam_param[1], left_cam_param[2]], 
                                        [left_cam_param[3], left_cam_param[4], left_cam_param[5]], 
                                        [left_cam_param[6], left_cam_param[7], left_cam_param[8]]])
            self.Left_dist_coeffs = np.array([[left_cam_param[9]], [left_cam_param[10]], [left_cam_param[11]], [left_cam_param[12]], [left_cam_param[13]]])

    
            self.Right_camera_matrix = np.array([[right_cam_param[0], right_cam_param[1], right_cam_param[2]], 
                                        [right_cam_param[3], right_cam_param[4], right_cam_param[5]], 
                                        [right_cam_param[6], right_cam_param[7], right_cam_param[8]]])
            self.Right_dist_coeffs = np.array([[right_cam_param[9]], [right_cam_param[10]], [right_cam_param[11]], [right_cam_param[12]], [right_cam_param[13]]])

            
            '''Mult-calibration parameters'''
            with open('./calibration/l_camera_Rt.txt', 'r') as f:
                for line in f.readlines():
                    left_cam_rt.extend([float(i) for i in line.split(',')])

            self.Left_camera_RT = np.array([[left_cam_rt[0], left_cam_rt[1], left_cam_rt[2], left_cam_rt[9]],
                                            [left_cam_rt[3], left_cam_rt[4], left_cam_rt[5], left_cam_rt[10]],
                                            [left_cam_rt[6], left_cam_rt[7], left_cam_rt[8], left_cam_rt[11]]])

            with open('./calibration/r_camera_Rt.txt', 'r') as f:
                for line in f.readlines():
                    right_cam_rt.extend([float(i) for i in line.split(',')])

            self.Right_camera_RT = np.array([[right_cam_rt[0], right_cam_rt[1], right_cam_rt[2], right_cam_rt[9]],
                                            [right_cam_rt[3], right_cam_rt[4], right_cam_rt[5], right_cam_rt[10]],
                                            [right_cam_rt[6], right_cam_rt[7], right_cam_rt[8], right_cam_rt[11]]])

            '''LiDAR-calibration parameters'''
            lidar_calib_param = []
            with open(LiDAR_cam_path, 'r') as f:
                for line in f.readlines():
                    lidar_calib_param.extend([float(i) for i in line.split(',')])

            self.lidar_RT = np.array([[lidar_calib_param[0], lidar_calib_param[1], lidar_calib_param[2], lidar_calib_param[9]],
                                    [lidar_calib_param[3], lidar_calib_param[4], lidar_calib_param[5], lidar_calib_param[10]],
                                    [lidar_calib_param[6], lidar_calib_param[7], lidar_calib_param[8], lidar_calib_param[11]]])
            
        else:
            print("Check for txt files")
        
        '''Front'''
        self.proj_lidar2cam = np.dot(self.camera_matrix, self.lidar_RT)

        # '''Side'''
        self.proj_lidar2cam_le = np.dot(self.Left_camera_matrix, self.Left_camera_RT)
        self.proj_lidar2cam_ri = np.dot(self.Right_camera_matrix, self.Right_camera_RT)
        
        roi_y = 3.0
        roi_x_top = 50.0
        roi_x_bottom = 4.0

        self.pt1_2 = [roi_x_top, roi_y]  # left top
        self.pt2_2 = [roi_x_top, -roi_y] # right top
        self.pt3_2 = [roi_x_bottom, -roi_y]  # right bottom 
        self.pt4_2 = [roi_x_bottom, roi_y]   # left bottom

        fix_z = [-1.62928751,-1.62468648, -1.68053416,-1.68513518]

        for i, xy in enumerate([self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2]):
            xy.append(fix_z[i])

        self.pts_3d = np.array([self.pt1_2, self.pt2_2, self.pt3_2, self.pt4_2])
        
        pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "front")
        left_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "left")
        right_pts_2d = self.project_3d_to_2d(self.pts_3d.transpose(), "right")

        src_pt, self.M = self.get_M_matrix(pts_2d)
        l_src_pt, self.LM = self.get_M_matrix(left_pts_2d)
        r_src_pt, self.RM = self.get_M_matrix(right_pts_2d)

        self.src_pt = src_pt
        self.l_src_pt = l_src_pt
        self.r_src_pt = r_src_pt
        
        '''Only for Topview'''
        self.resolution = 10 # 1pixel = 10m
        self.grid_size = (int((self.pt1_2[1] - self.pt2_2[1]) * 100 / self.resolution), int((self.pt1_2[0] - self.pt3_2[0]) * 100 / self.resolution))
        
        topview_dst_pt = np.float32([[0, 0],
                                    [self.grid_size[0], 0], 
                                    [self.grid_size[0], self.grid_size[1]],
                                    [0, self.grid_size[1]]])

        self.topview_M = cv2.getPerspectiveTransform(src_pt, topview_dst_pt)

    def project_3d_to_2d(self, points, task):
        num_pts = points.shape[1]
        points = np.vstack((points, np.ones((1, num_pts))))
        if task == "front":
            points = np.dot(self.proj_lidar2cam, points)
        elif task == "left":
            points = np.dot(self.proj_lidar2cam_le, points)
        elif task == "right":
            points = np.dot(self.proj_lidar2cam_ri, points)

        points[:2, :] /= points[2, :]
        return points[:2, :]

    def get_M_matrix(self, pts_2d):
        src_pt = np.float32([[int(np.round(pts_2d[0][0])), int(np.round(pts_2d[1][0]))],
                             [int(np.round(pts_2d[0][1])), int(np.round(pts_2d[1][1]))],
                             [int(np.round(pts_2d[0][2])), int(np.round(pts_2d[1][2]))],
                             [int(np.round(pts_2d[0][3])), int(np.round(pts_2d[1][3]))]])

        self.dst_pt = np.float32([[self.pt1_2[0], self.pt1_2[1]], 
                                [self.pt2_2[0], self.pt2_2[1]], 
                                [self.pt3_2[0], self.pt3_2[1]], 
                                [self.pt4_2[0], self.pt4_2[1]]])

        M2 = cv2.getPerspectiveTransform(src_pt, self.dst_pt)
        
        return src_pt, M2
        
    def undistort(self, img, task):
        w,h = (img.shape[1], img.shape[0])

        if task == "front":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        elif task == "left":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.Left_camera_matrix, self.Left_dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.Left_camera_matrix, self.Left_dist_coeffs, None, newcameramtx)

        elif task == "right":
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.Right_camera_matrix, self.Right_dist_coeffs, (w,h), 0)
            result_img = cv2.undistort(img, self.Right_camera_matrix, self.Right_dist_coeffs, None, newcameramtx)

        return result_img
        
    def topview(self, img):
        import math
        topveiw_img = cv2.warpPerspective(img, self.topview_M, (self.grid_size[0], self.grid_size[1]))
        return topveiw_img
        
