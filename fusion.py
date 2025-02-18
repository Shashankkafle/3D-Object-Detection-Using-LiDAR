import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kitti_utils import *

plt.rcParams["figure.figsize"] = (20, 10)


DATA_PATH = r'data'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# set confidence and IOU thresholds
model.conf = 0.25  # confidence threshold (0-1), default: 0.25
model.iou = 0.25  # NMS IoU threshold (0-1), default: 0.45
# get RGB camera data
image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02_end/images/*.png')))
# right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points_end/data/*.bin')))


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

def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    ''' Obtains detected object centers projected to uvz camera coordinates. 
        Starts by associating LiDAR uvz coordinates to detected object centers,
        once a match is found, the coordiantes are transformed to the uvz
        camera reference and added to the bboxes array.

        NOTE: The image is modified in place so there is no need to return it.

        Inputs:
          image - input image for detection 
          velo_uvz - LiDAR coordinates projected to camera reference
          bboxes - xyxy bounding boxes form detections from yolov5 model output
          draw - (_Bool) draw measured depths on image
        Outputs:
          bboxes_out - modified array containing the object centers projected
                       to uvz image coordinates
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = torch.round(bbox[0:2]).to(torch.int).numpy()
        pt2 = torch.round(bbox[2:4]).to(torch.int).numpy()

        # get center location of the object on the image
        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u)) 
                              - np.array([[obj_x_center, obj_y_center]]).T)
        
        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])
        
        # add velo projections (u, v, z) to bboxes_out
        bboxes_out[i, -3:] = uvz_location

        # draw depth on image at center of each bounding box
        # This is depth as perceived by the camera
        if draw:
            object_center = (np.round(obj_y_center).astype(int), 
                             np.round(obj_x_center).astype(int))
            cv2.putText(image, 
                        '{0:.2f} m'.format(velo_depth), 
                        object_center, # top left
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font scale
                        (255, 0, 0), 2, cv2.LINE_AA)    
            
    return bboxes_out

def get_detection_coordinates(image, bin_path, draw_boxes=True, draw_depth=True):
    ''' Obtains detections for the input image, along with the coordinates of 
        the detected object centers. The coordinate obtained are:
            - Camera with depth --> uvz 
            - LiDAR/velo --> xyz
            - GPS/IMU --> xyz
        Inputs:
            image - rgb image to run detection on
            bin_path - path to LiDAR bin file
        Output:
            bboxes - array of detected bounding boxes, confidences, classes,
            velo_uv - LiDAR points porjected to camera uvz coordinate frame
            coordinates - array of all object center coordinates in the frames
                          listed above
        '''
    ## 1. compute detections in the left image
    detections = model(image)

    # draw boxes on image
    if draw_boxes:
        detections.show() 

    # get bounding box locations (x1,y1), (x2,y2) Prob, class
    bboxes = detections.xyxy[0].cpu() # remove from GPU

    # get LiDAR points and transform them to image/camera space
    velo_uvz = project_velobin2uvz(bin_path, 
                                   T_velo_cam2, 
                                   image, 
                                   remove_plane=True)

    # get uvz centers for detected objects
    bboxes = get_uvz_centers(image, 
                             velo_uvz, 
                             bboxes, 
                             draw=draw_depth)

    return bboxes, velo_uvz




bin_path = bin_paths[0]  

image = cv2.imread(image_paths[0])



bboxes, velo_uvz = get_detection_coordinates(image, bin_path)
uvz = bboxes[:, -3:]

def visualize_velo_uvz(velo_uvz):
    """
    Visualize the LiDAR points in image space using Matplotlib.
    
    Args:
        velo_uvz: Nx3 array of LiDAR points in camera coordinates (u, v, z)
    """
    # Extract u, v, z coordinates
    u = velo_uvz[:, 0]
    v = velo_uvz[:, 1]
    z = velo_uvz[:, 2]

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(u, v, c=z, cmap='viridis', s=1)  # Color by depth (z)
    plt.colorbar(label='Depth (z)')
    plt.xlabel('u (image x-coordinate)')
    plt.ylabel('v (image y-coordinate)')
    plt.title('LiDAR Points in Image Space')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

# Example usage
# Assuming velo_uvz is already defined
# velo_uvz = project_velobin2uvz(bin_path, T_velo_cam2, image, remove_plane=True)
visualize_velo_uvz(velo_uvz)
print("velo_uvz", velo_uvz)

# draw velo on blank image
velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(image))

plt.imshow(velo_image);

# stack frames
stacked = np.vstack((image, velo_image))


canvas_height = stacked.shape[0]
canvas_width = 500

# get consistent center for ego vehicle
ego_center = (250, int(canvas_height*0.95))

# get rectangle coordiantes for ego vehicle
ego_x1 = ego_center[0] - 5
ego_y1 = ego_center[1] - 10
ego_x2 = ego_center[0] + 5
ego_y2 = ego_center[1] + 10



# draw top down scenario on canvas
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)



def draw_scenario(canvas, imu_xyz, sf=12):
    # draw ego vehicle
    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (0, 255, 0), -1);

    # draw detected objects
    for val in imu_xyz:
        obj_center = (ego_center[0] - sf*int(np.round(val[1])),
                      ego_center[1] - sf*int(np.round(val[0])))
        # cv2.circle(canvas, obj_center, 5, (255, 0, 0), -1);

        # get object rectangle coordinates
        obj_x1 = obj_center[0] - 5
        obj_y1 = obj_center[1] - 10
        obj_x2 = obj_center[0] + 5
        obj_y2 = obj_center[1] + 10

        cv2.rectangle(canvas, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 0), -1);


    return canvas



# Display the image
cv2.imshow('Stacked image', stacked)

# Wait for a key press to close the window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()