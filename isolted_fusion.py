import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kitti_utils import *

plt.rcParams["figure.figsize"] = (20, 10)


DATA_PATH = r'data'

image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02_end/images/*.png')))
# right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points_end/data/*.bin')))

# Visualize the LiDAR points before projection
# Load LiDAR data
xyzw = bin2xyzw(bin_paths[2], remove_plane=False)

# Visualize all LiDAR points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyzw[:, 0], xyzw[:, 1], xyzw[:, 2], c='r', s=1)  # Plot LiDAR points in red
ax.set_title("3D LiDAR Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

print(f"Number of left images: {len(image_paths)}")
# print(f"Number of right images: {len(right_image_paths)}")
print(f"Number of LiDAR point clouds: {len(bin_paths)}")
# print(f"Number of GPS/IMU frames: {len(oxts_paths)}")

with open('data/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# get projection matrices (rectified left camera --> left camera (u,v,z))
P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))


# get rectified rotation matrices (left camera --> rectified left camera)
R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))

# add (0,0,0) translation and convert to homogeneous coordinates
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)


# get rigid transformation from Camera 0 (ref) to Camera 2
R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))

# get cam0 to cam2 rigid body transformation in homogeneous coordinates
T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)


# Load LiDAR Calibration Data
T_velo_ref0 = get_rigid_transformation(r'data/calib_velo_to_cam.txt')


# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 


print("T_velo_cam2", T_velo_cam2)
image = cv2.imread(image_paths[0])

projected_lidar = project_velobin2uvz(bin_paths[0], T_velo_cam2,image=image,remove_plane=False)
print(projected_lidar)
print("projected_lidar.shape",projected_lidar.shape)
print("image shape",image.shape)






# Display the image with projected LiDAR points
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(projected_lidar[:, 0], projected_lidar[:, 1], c='b', s=1)  # Plot LiDAR points
plt.title("LiDAR Projection on Image")
plt.xlabel("Image Width")
plt.ylabel("Image Height")
plt.show()






