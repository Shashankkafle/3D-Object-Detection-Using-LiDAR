import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams["figure.figsize"] = (20, 10)


def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    ''' Converts LiDAR points to camera uvz and associates them with detected bounding boxes.
        Inputs:
            image - the RGB image
            velo_uvz - LiDAR points in camera coordinate frame (x, y, z) or (u, v, z)
            bboxes - bounding boxes with detected object information
            draw - whether to draw the projected centers on the image
        Output:
            bboxes - updated bounding boxes with associated uvz coordinates
    '''
    # Extract the image size for later use (to handle uvz -> pixel mapping)
    height, width = image.shape[:2]

    # Process each bounding box and compute center in LiDAR frame
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, prob, class_id = bbox

        # Compute the center of the bounding box in pixel space
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Get the corresponding LiDAR points in the center of the bounding box
        # Assuming `velo_uvz` is an array of LiDAR points with (u, v, z)
        # You would use some method to map from camera coordinates to LiDAR frame
        lidar_point = velo_uvz[int(center_y), int(center_x)]  # this assumes velo_uvz is structured similarly
        
        # Convert to depth (z) and pixel coordinates (u, v)
        u, v, z = lidar_point  # Assuming velo_uvz is in (u, v, z) format
        
        # Update bbox with this center's uvz information
        bboxes[i] = np.append(bboxes[i], [u, v, z])  # Append uvz to bbox information

        if draw:
            # Optionally, draw the center of the bounding box on the image
            cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)  # draw center

    return bboxes



def get_rigid_transformation(file_path):
    ''' Loads a rigid transformation matrix from a calibration file. The file contains 
        rotation and translation components for transforming from one coordinate frame to another.
        
        Inputs:
            file_path - path to the calibration file (e.g., calib_velo_to_cam.txt)
        
        Output:
            transformation_matrix - 4x4 transformation matrix (rotation + translation)
    '''
    print("path",file_path)
    # Read the transformation from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Typically, the last line in the file contains the transformation matrix in row-major format
    # assuming the file format looks like:
    # R: <rotation matrix>
    # T: <translation vector>
    
    # Find the line that contains the transformation and extract it
    transformation = np.zeros((4, 4))  # Initialize a 4x4 matrix
    
    for i, line in enumerate(lines):
        if "R:" in line:
            # Extract the rotation matrix (assuming it's the next few lines after "R:")
            rotation = np.array([list(map(float, lines[i+1].strip().split()))])
            transformation[:3, :3] = rotation.reshape(3, 3)
        elif "T:" in line:
            # Extract the translation vector (assuming it's the next line after "T:")
            translation = np.array(list(map(float, lines[i+1].strip().split())))
            transformation[:3, 3] = translation
        else:
            continue
    
    # For a rigid transformation, the last row should be [0, 0, 0, 1]
    transformation[3, 3] = 1.0
    
    return transformation


DATA_PATH = r''

# get RGB camera data
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'data/image_02/images_start/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'data/velodyne_points_start/data/*.bin')))



print(f"Number of left images: {len(left_image_paths)}")
print(f"Number of LiDAR point clouds: {len(bin_paths)}")

# Load calibration data
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


T_velo_ref0 = get_rigid_transformation(r'data/calib_velo_to_cam.txt')




# Obtain matrix to transform 3D LiDAR/velo (x, y, z) coordiantes to 2D camera (u,v) coordinates, and it's homogeneous inverse that will allow us to transform from camera (u, v, z, 1) back to LiDAR (x, y, z, 1)

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0))



