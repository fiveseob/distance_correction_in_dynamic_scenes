# CVLab_Project
## Index
  - [Overview](#overview) 
  - [Getting Started](#getting-started)
  - [Contributing](#contributing)
  - [Authors](#authors)
  - [License](#license)
  
  ```sh
  ├── README.md
  ├── Optical.py                  - OpticalFlow(Simulator)
  ├── pose_esti_plot.py           - Compare Plot(Pose)
  ├── esti_dist_RW.py             - Distance Estimation & Correction (RealWorld)
  ├── esti_dist_SM.py             - Distance Estimation & Correction (Simulator)
  ├── distance_diff_plot.py       - Compare Plot(Dist)     
  ├── API      
  │   ├── drawer.py               - Drawing on Output result Image 
  │   ├── tracker.py              - Tracking (Not using)    
  ├── data
  │   ├── 2021_coco.names         - Simulator version
  │   ├── coco.names              - coco 80 version
  ├── models
  │   ├── Yolov4_model.py         - python3 
  │   │── py2_Yolov4_model.py     - python2
  ├── tool
  │   ├── region_loss.py
  │   ├── yolo_loss.py
  │   ├── utils.py
  │   ├── torch_utils.py
  │   └── yolo_layer.py
  ```
  
<!--  Other options to write Readme
  - [Deployment](#deployment)
  - [Used or Referenced Projects](Used-or-Referenced-Projects)
-->
## About RepositoryTemplate
<!--Wirte one paragraph of project description -->  
The purpose of this project is to proceed with the CVLab project.

## Overview
<!-- Write Overview about this project -->
**If you use this template, you can use this function**
- Calibration
- Pose Estimation
- Pose Correction
- Distance Estimation

## Getting Started
**click `Use this template` and use this template!**
<!--
### Depencies
 Write about need to install the software and how to install them 
-->
### Environment
 - Pytorch 1.6
 - CUDA 10.2/11.0
 - cuDNN 7.6.5
 - ROSVERSION 1.14.5
 - tensorRT 7.0.0.11
 - OPENCV 3.2.0
 - scipy 1.2.3
 - python 2.7.17
 
<!--
## Deployment
 Add additional notes about how to deploy this on a live system
 -->
## Contributing
<!-- Write the way to contribute -->
I am looking for someone to help with this project. Please advise and point out.  

## Authors

See also the list of [contributors](https://github.com/always0ne/readmeTemplate/contributors)
who participated in this project.
<!--
## Used or Referenced Projects
 - [referenced Project](project link) - **LICENSE** - little-bit introduce
-->

## License
```

```# Distance correction in dynamic scenes
# distance_correction_in_dynamic_scenes
