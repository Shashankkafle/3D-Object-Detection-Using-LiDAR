import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_lidar_bin(file_path):
    """Load LiDAR .bin file assuming (x, y, z, intensity) float32 format."""
    lidar_data = np.fromfile(file_path, dtype=np.float32)
    lidar_data = lidar_data.reshape(-1, 4)  # Assuming each point has (x, y, z, intensity)
    return lidar_data

def visualize_matplotlib(lidar_data):
    """Visualize LiDAR point cloud using Matplotlib."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], 
               c=lidar_data[:, 3], cmap='jet', s=0.1)  # Color by intensity
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Point Cloud (Matplotlib)')
    
    plt.show()

def visualize_open3d(lidar_data):
    """Visualize LiDAR point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])  # Use only x, y, z
    
    # Optional: Color by intensity (normalized)
    intensity = lidar_data[:, 3]
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())  # Normalize
    colors = np.column_stack((intensity, intensity, intensity))  # Grayscale color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name="Open3D LiDAR Visualization")

if __name__ == "__main__":
    print("miain")
    # parser = argparse.ArgumentParser(description="Visualize LiDAR .bin file.")
    # parser.add_argument("file", type=str, help="Path to the .bin LiDAR file")
    # args = parser.parse_args()
    # print("args", args)
    # Load LiDAR data
    lidar_points = load_lidar_bin("/Users/shashankkafle/Documents/SIIT Masters/Lidar_camera_fusion/data/velodyne_points_start/data/0000000000.bin")
    print("lidar_points", lidar_points)
    # Visualize using Matplotlib
    visualize_matplotlib(lidar_points)
    
    # Visualize using Open3D
    visualize_open3d(lidar_points)
