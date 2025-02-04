import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kitti_utils import *

plt.rcParams["figure.figsize"] = (20, 10)


DATA_PATH = r'data'



# get RGB camera data
image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02_start/images/*.png')))
# right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points_start/data/*.bin')))


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

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 

def transform_uvz(uvz, T):
    """
    Transform coordinates from camera (u,v,z) to another coordinate system using transformation matrix T
    
    Args:
        uvz: Nx3 array of coordinates in camera frame (u,v,z)
        T: 4x4 homogeneous transformation matrix
    
    Returns:
        Nx3 array of transformed coordinates (x,y,z)
    """
    # Convert to homogeneous coordinates
    uvz_h = np.hstack((uvz, np.ones((uvz.shape[0], 1))))
    
    # Transform coordinates
    xyz_h = (T @ uvz_h.T).T
    
    # Convert back from homogeneous coordinates
    xyz = xyz_h[:, :3] / xyz_h[:, 3:]
    
    return xyz

def draw_velo_on_image(velo_uvz, img):
    """
    Draw LiDAR points on image
    
    Args:
        velo_uvz: Nx3 array of LiDAR points in camera coordinates (u,v,z)
        img: Image to draw points on
    
    Returns:
        Image with LiDAR points drawn on it
    """
    # Create copy of image
    img_copy = img.copy()
    
    # Get image dimensions
    img_h, img_w = img.shape[:2]
    
    # Filter points that are behind the camera
    mask = velo_uvz[:, 2] > 0
    velo_uvz = velo_uvz[mask]
    
    # Convert to integer pixel coordinates
    u = velo_uvz[:, 0].astype(np.int32)
    v = velo_uvz[:, 1].astype(np.int32)
    
    # Filter points that are outside the image
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[mask]
    v = v[mask]
    
    # Get depth for coloring
    depth = velo_uvz[mask][:, 2]
    
    # Color points based on depth
    colors = depth_to_colors(depth)
    
    # Draw points
    for i in range(len(u)):
        cv2.circle(img_copy, (u[i], v[i]), 2, colors[i].tolist(), -1)
    
    return img_copy

def depth_to_colors(depth):
    """
    Convert depth values to colors
    
    Args:
        depth: Array of depth values
    
    Returns:
        Array of RGB colors
    """
    # Normalize depth values to 0-1 range
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Convert to colors (red=near, blue=far)
    colors = np.zeros((len(depth), 3))
    colors[:, 0] = 255 * (1 - depth_normalized)  # Red channel
    colors[:, 2] = 255 * depth_normalized        # Blue channel
    
    return colors


canvas_height = 720  
canvas_width = 1280  

bin_path = bin_paths[0]  

image = cv2.imread(image_paths[0])

# get LiDAR points and transform them to image/camera space
velo_uvz = project_velobin2uvz(bin_path, T_velo_cam2, image, remove_plane=True)

# draw velo on blank image
velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(image))
# stack frames
stacked = np.vstack((image, velo_image))
# draw top down scenario on canvas
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Display the image
plt.rcParams["figure.figsize"] = (20, 10)

# stack image with LiDAR point cloud
stacked = np.vstack((image, velo_image))

# display stacked iamge
plt.imshow(stacked);# # Wait for a key press to close the window
# cv2.waitKey(0)

# # Close all OpenCV windows
# cv2.destroyAllWindows()