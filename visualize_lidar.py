import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob  # Add this line to import the glob module
from kitti_utils import *

def visualize_lidar_points(lidar_data):
    """Visualize raw LiDAR points in 3D."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Raw LiDAR Points')
    plt.show()

def visualize_projected_points(velo_uvz):
    """Visualize projected LiDAR points in 2D."""
    plt.figure(figsize=(10, 7))
    plt.scatter(velo_uvz[:, 0], velo_uvz[:, 1], c=velo_uvz[:, 2], cmap='viridis', s=1)
    plt.colorbar(label='Depth (z)')
    plt.xlabel('u (image x-coordinate)')
    plt.ylabel('v (image y-coordinate)')
    plt.title('Projected LiDAR Points')
    plt.gca().invert_yaxis()
    plt.show()

DATA_PATH = r'data'

bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points_end/data/*.bin')))
print(bin_paths[0])
# Example usage
lidar_data = np.fromfile(bin_paths[0], dtype=np.float32).reshape(-1, 4)
visualize_lidar_points(lidar_data)

# Assuming velo_uvz is the result of your projection function
# visualize_projected_points(velo_uvz)