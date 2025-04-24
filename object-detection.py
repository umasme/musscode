import open3d as o3d
import numpy as np
import pyrealsense2 as rs

def generatePCD(camera, frames):
    depth_frame = frames[camera].get_depth_frame()
    if depth_frame:
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        # Extract vertices and colors
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # 3D points
        colors = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # Texture mapping

        # Normalize colors
        colors_rgb = []
        for tex_coords in colors:
            u, v = tex_coords[0], tex_coords[1]
            if 0 <= u < 1 and 0 <= v < 1:  # Ensure valid texture coordinates
                x = int(u * color_frame.get_width())
                y = int(v * color_frame.get_height())
                colors_rgb.append(color_image[y, x] / 255.0)  # Normalize to 0-1
            else:
                colors_rgb.append([0, 0, 0])  # Default color for invalid texture coordinates

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        return pcd



def checkForObstacles(camera, frames):
    pcd = generatePCD(camera, frames)
    points = np.asarray(pcd.points)

    # Downsample the point cloud for efficiency
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Perform plane segmentation using RANSAC to find the ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                             ransac_n=3,
                                             num_iterations=1000)
    a, b, c, d = plane_model

    # Extract outliers (non-ground points)
    non_ground_cloud = pcd.select_by_index(inliers, invert=True)

    # Filter out points that are too high or low above the ground plane
    # Assuming a maximum height of 1 meter above the ground plane for obstacle detection
    min_height_meters = 0.1
    max_height_meters = 1.0
    non_ground_cloud_points = np.asarray(non_ground_cloud.points)
    distances_to_plane = (a * non_ground_cloud_points[:, 0] + b * non_ground_cloud_points[:, 1] + c * non_ground_cloud_points[:, 2] + d)
    valid_height_indices = np.where((distances_to_plane <= max_height_meters) & (distances_to_plane >= min_height_meters))[0]
    non_ground_cloud = non_ground_cloud.select_by_index(valid_height_indices)
    non_ground_cloud_points = np.asarray(non_ground_cloud.points)

    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector

    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    camera_forward = np.array([0, 0, 1])  # Assuming camera is looking in the z direction

    #Project camera forward derection onto the ground plane
    forward_projected = camera_forward - np.dot(camera_forward, normal_vector) * normal_vector
    if np.linalg.norm(forward_projected) < 1e-6:
        x_axis = np.array([1, 0, 0])
        if abs(np.dot(x_axis, normal_vector)) > 0.9:
            x_axis = np.array([0, 1, 0])
        x_axis = x_axis - np.dot(x_axis, normal_vector) * normal_vector
    
    x_parallel = forward_projected / np.linalg.norm(forward_projected)  # Normalize the projected vector
    y_parallel = np.cross(normal_vector, x_parallel)  # Cross product to get the y-axis
    y_parallel /= np.linalg.norm(y_parallel)  # Normalize the y-axis
    
    #2D projection of the point cloud onto the ground plane
    projection_matrix = np.column_stack((x_parallel, y_parallel))
    points_2d = np.dot(points, projection_matrix.T)  # Project points onto the ground plane
    rectangle_length = 5.0  # Length of the rectangle in meters
    rectangle_width = 0.9  # Width of the rectangle in meters

    #Check if within rectangle
    in_rectangle = np.logical_and(
        np.logical_and(points_2d[:, 0] >= 0, points_2d[:, 0 <= rectangle_length]),
                        np.logical_and(points_2d[:, 1] >= -rectangle_width/2, points_2d[:, 1] <= rectangle_width/2))
    
    obstacle_points = points[in_rectangle]
    if len(obstacle_points) > 0:
        distances = np.dot(obstacle_points, x_parallel)  # Project points onto the x-axis
        min_distance = np.min(distances)
        print(f"Closest obstacle is {min_distance:.2f} meters away")
        return min_distance
    else:
        print("No obstacles detected")
        return None

realsense_pipelines = []
realsense_profiles = []
ctx = rs.context()
realSense_devices = ctx.query_devices()

for cam_idx in range(2):  # For two RealSense cameras
            pipeline = rs.pipeline()
            config = rs.config()
            serial_number = realSense_devices[cam_idx].get_info(rs.camera_info.serial_number)
            print(f"Starting camera with serial number: {serial_number}")
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = pipeline.start(config)
            realsense_pipelines.append((pipeline, serial_number))
            realsense_profiles.append(profile)

frames = {}
for pipeline, serial_number in realsense_pipelines:
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if color_frame:
        color_image = np.asanyarray(color_frame.get_data())
        stream_name = f"realsense-{serial_number}"
        mxId = f"realsense-{serial_number}"  # Fake ID, update CAMERA_INFOS if needed
        frames.update({mxId : frame}) # Store the frame for later use

obstacle_range = checkForObstacles("realsense-247122073398", frames)  # Replace with actual camera ID
print(f"Obstacle range: {obstacle_range}")

# Stop the pipelines after use
for pipeline, _ in realsense_pipelines:
    pipeline.stop()
