from calibration.calib import Calibration

import ROS_tools
import pose_estimation
if __name__ == "__main__":
    camera_path = ['calibration/f_camera_1280.txt', 'calibration/l_camera_1280.txt', 'calibration/r_camera_1280.txt']
    imu_camera_path = './calibration/camera_imu.txt'
    LiDAR_camera_path = 'calibration/f_camera_lidar_1280.txt'

    calib = Calibration(camera_path, imu_camera_path , LiDAR_camera_path)

    Pose = pose_estimation.PoseEstimation(calib)
    Pose.main()
