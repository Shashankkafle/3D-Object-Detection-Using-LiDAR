import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from kitti_utils import *  # Assuming kitti_utils.py is in the same directory




# Set up matplotlib
plt.rcParams["figure.figsize"] = (20, 10)

# Download KITTI data (if not already downloaded)
if not os.path.exists('2011_10_03_drive_0047_sync.zip'):
    os.system('wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip')
if not os.path.exists('2011_10_03_calib.zip'):
    os.system('wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip')

# Extract the downloaded zip files
os.system('jar xf 2011_10_03_drive_0047_sync.zip')
os.system('jar xf 2011_10_03_calib.zip')

# Define utility functions
def get_total_seconds(hms):
    return hms[0] * 60 * 60 + hms[1] * 60 + hms[2]

def timestamps2seconds(timestamp_path):
    ''' Reads in timestamp path and returns total seconds (does not account for day rollover) '''
    timestamps = pd.read_csv(timestamp_path, header=None, squeeze=True).astype(object).apply(lambda x: x.split(' ')[1])
    hours = timestamps.apply(lambda x: x.split(':')[0]).astype(np.float64)
    minutes = timestamps.apply(lambda x: x.split(':')[1]).astype(np.float64)
    seconds = timestamps.apply(lambda x: x.split(':')[2]).astype(np.float64)
    hms_vals = np.vstack((hours, minutes, seconds)).T
    total_seconds = np.array(list(map(get_total_seconds, hms_vals)))
    return total_seconds

# Define paths
DATA_PATH = '2011_10_03_drive_0047_sync'  # Adjust this path if necessary
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
oxts_paths = sorted(glob(os.path.join(DATA_PATH, 'oxts/data/*.txt')))

# Calculate camera 2 frames per second
cam2_total_seconds = timestamps2seconds(os.path.join(DATA_PATH, 'image_02/timestamps.txt'))
cam2_fps = 1 / np.median(np.diff(cam2_total_seconds))

# Process each frame
result_video = []
for index in range(len(left_image_paths)):
    left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    bin_path = bin_paths[index]
    oxts_frame = get_oxts(oxts_paths[index])

    # Get detections and object centers in uvz
    bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path)

    # Get transformed coordinates
    uvz = bboxes[:, -3:]
    imu_xyz = transform_uvz(uvz, T_cam2_imu)

    # Draw velo on blank image
    velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))

    # Stack frames
    stacked = np.vstack((left_image, velo_image))

    # Draw top down scenario on canvas
    canvas_height, canvas_width = 720, 1440  # Adjust these dimensions as needed
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    draw_scenario(canvas, imu_xyz, sf=12)

    # Place everything in a single frame
    frame = np.hstack((stacked, 255 * np.ones((canvas_height, 1, 3), dtype=np.uint8), canvas))

    # Add to result video
    result_video.append(frame)

# Get width and height for video frames
h, w, _ = frame.shape

# Write video
out = cv2.VideoWriter('lidar_frame_stack.avi', cv2.VideoWriter_fourcc(*'DIVX'), cam2_fps, (w, h))
for i in range(len(result_video)):
    out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
out.release()

# Display the last frame
plt.imshow(frame)
plt.show()