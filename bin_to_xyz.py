import numpy as np

def bin_to_xyz(bin_file, output_file):
    # Read the binary file
    points = np.fromfile(bin_file, dtype=np.float32)
    
    # Reshape the data based on the number of fields per point (e.g., (x, y, z, intensity) -> 4 fields per point)
    # Assuming each point is represented by 4 floats (x, y, z, intensity)
    points = points.reshape((-1, 4))
    
    # Extract x, y, z coordinates
    xyz_points = points[:, :3]  # Taking only the x, y, z columns
    
    # Write to output file in .xyz format
    np.savetxt(output_file, xyz_points, fmt='%.6f', delimiter=' ')

# Example usage:
bin_file = '/Users/shashankkafle/Documents/SIIT Masters/Fusion With Object Detection/2011_10_03/2011_10_03_drive_0047_sync/velodyne_points/data/0000000000.bin'
output_file = 'output_file.xyz'
bin_to_xyz(bin_file, output_file)
