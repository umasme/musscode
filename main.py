#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import contextlib
import time
import pyrealsense2 as rs  # Added RealSense library
from pysabertooth import Sabertooth
import open3d as o3d

''' TODO:
1. check theta returned from aruco detection is accurate
2. add intrinsic parameters for all four cameras
3. test with all four cameras
4. add code for encoders when aruco is not detected
5. add object detection
6. add object detection behavior
7. add object detection behavior to waypoints
8. 
'''
# Camera intrinsic parameters (from Matlab) for (left or right) oak-d-lite
fx1 = 1515.24261837315  # Focal length x
fy1 = 1513.21547841726  # Focal length y
cx1 = 986.009156502993   # Optical center x
cy1 = 551.618039270305   # Optical center y
camera_matrix1 = np.array([[fx1, 0, cx1],
                           [0, fy1, cy1],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs1 = np.array((0.114251294509202,-0.228889968220235,0,0))

relative_position1 = [0.145, 0, 180] # Relative position of the camera with respect to the robot base (X, Y, Theta)
relative_position2 = [-0.145, 0, 0]  # Relative position of the camera with respect to the robot base (X, Y, Theta)

# Camera intrinsic parameters for the other (left or right) oak-d-lite (assumed the same for both cameras for now (03/18/25) update when other cameras are calibrated)
fx2 = 1515.24261837315  # Focal length x
fy2 = 1513.21547841726  # Focal length y
cx2 = 986.009156502993   # Optical center x
cy2 = 551.618039270305   # Optical center y
camera_matrix2 = np.array([[fx2, 0, cx2],
                           [0, fy2, cy2],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs2 = np.array((0.114251294509202,-0.228889968220235,0,0))

# Default camera intrinsic parameters for RealSense D435i (assumed the same for both cameras for now (03/18/25) update when other cameras are calibrated)
camera_matrix3 = np.array([[1384, 0, 960],
                           [0, 1384, 540],
                           [0, 0, 1]], dtype=float) 

dist_coeffs3 = np.array((0, 0, 0, 0))

relative_position3 = [0, 0.145, 90] # Relative position of the camera with respect to the robot base (X, Y, Theta)  
relative_position4 = [0, -0.145, 270] # Relative position of the camera with respect to the robot base (X, Y, Theta)


CAMERA_INFOS = {
 "14442C10911DC5D200" : {"camera_matrix" : camera_matrix1, "dist_coeffs" : dist_coeffs1, "relative_position" : relative_position1},
 "14442C1071EDDFD600" : {"camera_matrix" : camera_matrix2, "dist_coeffs" : dist_coeffs2, "relative_position" : relative_position2},
 "realsense-247122073398": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position3},
 "realsense-327122073351": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position4},
}

WAYPOINTS = [[1, -2, "mine"], [1.5, -2, "deposit"], [1, -1, "deposit"]]  # Updated waypoints

marker_size = 0.11

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def createPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    camRgb.setPreviewSize(640, 480)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)

    # Create output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)

    xlinkOut.setStreamName("imu")

    # Enable ACCELEROMETER_RAW at 500 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
    # Enable GYROSCOPE_RAW at 400 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

    # Set batch report thresholds
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Link plugins IMU -> XLINK
    imu.out.link(xlinkOut.input)

    return pipeline

def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds() * 1000

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

def localize(color_images, imuQueue, aruco_detector, marker_size, baseTs, prev_gyroTs, camera_position, pose):
    last_print_time = time.time()  # Initialize time tracking

    imuData = imuQueue.get()  # Blocking call, will wait until new data has arrived
    imuPackets = imuData.packets
    for color_image, stream_name, mxId in color_images:
        # Convert to grayscale for ArUco detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray_image)

        camera_matrix = CAMERA_INFOS[str(mxId)]["camera_matrix"]
        dist_coeffs = CAMERA_INFOS[str(mxId)]["dist_coeffs"]

        # Process each detected marker and get pose relative to id 2
        if ids is not None and 2 in ids:
            arr = np.where(ids == 2)[0][0]
            corners = np.array(corners[arr])
            ids = np.array(ids[arr])
            # Get the center of the marker
            center = np.mean(corners[0], axis=0).astype(int)

            rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            rvec = np.array(rvec)
            tvec = np.array(tvec)

            # Draw the marker and axes
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])  # Convert rotation vector to rotation matrix using the rodrigues formula
            R_inv = rotation_matrix.T  # Inverse of the rotation matrix
            camera_position = R_inv @ tvec[0]  # @ is the matrix multiplication operator in Python
            theta = np.arcsin(-R_inv[2][0])
            theta = np.degrees(theta)  # Convert radians to degrees for readability

            pose = [camera_position[0][0], camera_position[2][0], theta]  # Pose in the format [x, y, theta]
            pose = pose + np.array(CAMERA_INFOS[str(mxId)]["relative_position"])  # Adjust pose based on relative position of the camera
            pose[2] = pose[2] % 360  # Normalize theta to be between 0 and 360 degrees

        elif camera_position is not None:
            for imuPacket in imuPackets:
                acceleroValues = imuPacket.acceleroMeter
                gyroValues = imuPacket.gyroscope

                acceleroTs = acceleroValues.getTimestampDevice()
                gyroTs = gyroValues.getTimestampDevice()

                if baseTs is None:
                    baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
                    prev_gyroTs = gyroTs
                    print(baseTs)

                acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

                imuF = "{:.06f}"
                tsF = "{:.03f}"

                # Calculate the time difference between the current and previous gyroscope readings
                dt = (gyroTs - timeDeltaToMilliS(prev_gyroTs - baseTs)) / 1000.0  # Convert milliseconds to seconds
                prev_gyroTs = gyroValues.getTimestampDevice()

                gyroValues = round(gyroValues.x, 2), round(gyroValues.y, 2), round(gyroValues.z, 2)

                # Integrate the gyroscope data to get the angles
                pose[2] += np.degrees(gyroValues[1]) * dt  # Pitch

        current_time = time.time()
        if current_time - last_print_time >= 1 and camera_position is not None:
            print(f"Camera Position: {pose}")
            last_print_time = current_time  # Update last print time

        # Display the output image
        cv2.imshow(stream_name, color_image)

    return pose, baseTs, prev_gyroTs, camera_position

def turn_to(theta):
    if pose[2] - theta > 180:
         turn_left(50)
         print(f"Turning left to {theta}")
    elif pose[2] - theta < 180:
         turn_right(50)  # Adjust speed as necessary for turning, 20 is an example speed
         print(f"Turning right to {theta}")

def move_to(current_position, target_position):
    theta = np.degrees(np.atan2(target_position[1] - current_position[1], target_position[0] - current_position[0]))
    theta -= 90  # Adjust for camera orientation
    theta = theta % 360  # Normalize theta to be between 0 and 360 degrees
    if abs(current_position[2] - theta) > 5:
        turn_to(theta)
    else:
        linear_motion(20)
        print("Moving forward")


def excavate(initial_time):
    excavate_time = 5  # Duration of excavation in seconds
    if time.time() - initial_time < excavate_time:
        print("Excavating")
        return False  # Excavation is still in progress
    else: 
        print("Excavation complete")
        return True 

def deposit(initial_time):
    deposit_time = 5
    if time.time() - initial_time < deposit_time:
        print("Depositing")
        motor3.drive(1, 50)	# drive deposition motor
        return False
    else:
        print("Deposit complete")
        stop_all()
        return True
    
def stop_all():
	motor1.stop()			# Turn off both motors
	motor2.stop()	

def linear_motion(speed:int):
	## Motor 1
	motor1.drive(1,speed)	# Turn on motor 1
	motor1.drive(2,speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1, -speed)	# Turn on motor 1
	motor2.drive(2, -speed)	# Turn on motor 2

def turn_left(speed:int):
	
	## Motor 1
	motor1.drive(1,-speed)	# Turn on motor 1
	motor1.drive(2,speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1,-speed)	# Turn on motor 1
	motor2.drive(2,speed)	# Turn on motor 2

def turn_right(speed:int):
    	## Motor 1
	motor1.drive(1, speed)	# Turn on motor 1
	motor1.drive(2,-speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1,speed)	# Turn on motor 1
	motor2.drive(2,-speed)	# Turn on motor 2


motor1 = Sabertooth("/dev/serial0", baudrate = 9600, address = 129)	# Init the Motor
motor1.open()								# Open then connection
print(f"Connection Status: {motor1.saber.is_open}")			# Let us know if it is open
motor1.info()								# Get the motor info


## Init up the sabertooth 2, and open the seral connection 
motor2 = Sabertooth("/dev/serial0", baudrate = 9600, address = 134)	# Init the Motor
motor2.open()								# Open then connection
print(f"Connection Status: {motor2.saber.is_open}")			# Let us know if it is open
motor2.info()								# Get the motor info

motor3 = Sabertooth("/dev/serial0", baudrate = 9600, address = 128)	# Init the Motor
motor3.open()								# Open then connection
print(f"Connection Status: {motor3.saber.is_open}")			# Let us know if it is open
motor3.info()								# Get the motor info

try:
    with contextlib.ExitStack() as stack:
        deviceInfos = dai.Device.getAllAvailableDevices()
        usbSpeed = dai.UsbSpeed.SUPER
        openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

        qRgbMap = []
        devices = []

        realsense_pipelines = []
        realsense_profiles = []

        ctx = rs.context()
        realSense_devices = ctx.query_devices()

        for deviceInfo in deviceInfos:
            deviceInfo: dai.DeviceInfo
            device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
            devices.append(device)
            print("===Connected to ", deviceInfo.getMxId())
            mxId = device.getMxId()
            cameras = device.getConnectedCameras()
            usbSpeed = device.getUsbSpeed()
            eepromData = device.readCalibration2().getEepromData()
            print("   >>> MXID:", mxId)
            print("   >>> Num of cameras:", len(cameras))
            print("   >>> USB speed:", usbSpeed)
            if eepromData.boardName != "":
                print("   >>> Board name:", eepromData.boardName)
            if eepromData.productName != "":
                print("   >>> Product name:", eepromData.productName)

            pipeline = createPipeline()
            device.startPipeline(pipeline)

            # Output queue for imu bulk packets
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            # Output queue will be used to get the rgb frames from the output defined above
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            stream_name = "rgb-" + mxId + "-" + eepromData.productName
            qRgbMap.append((q_rgb, stream_name, mxId))

            # Create resizable windows for each stream
            # cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL)

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

        last_print_time = time.time()

        mining = False
        depositing = False

        baseTs = None
        prev_gyroTs = None
        pose = None  # Initialize pose as a list with [x, y, theta]
        camera_position = None

        i = 0
        waypoint = 0
        at_waypoint = False  # Flag to indicate if the robot has reached the current waypoint
        initially_turning = True  # Flag to indicate if the robot is initially turning

    

        while True:
            
            color_images = []
            frames = {}
            for pipeline, serial_number in realsense_pipelines:
                frame = pipeline.wait_for_frames()
                color_frame = frame.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    stream_name = f"realsense-{serial_number}"
                    mxId = f"realsense-{serial_number}"  # Fake ID, update CAMERA_INFOS if needed
                    color_images.append((color_image, stream_name, mxId))
                    frames.update({mxId : frame}) # Store the frame for later use
            
            for q_rgb, stream_name, mxId in qRgbMap:
                if q_rgb.has():
                    color_image = q_rgb.get().getCvFrame()
                    color_images.append((color_image, stream_name, mxId))

            # Pass all required arguments to the localize function
            pose, baseTs, prev_gyroTs, camera_position = localize(
                color_images, imuQueue, aruco_detector, marker_size, baseTs, prev_gyroTs, camera_position, pose
            )
            
            if pose is None:
                turn_left(20)  # If pose is None, rotate to find ArUco markers
                print("rotating to find ArUco markers...")


            elif pose is not None:
                current_time = time.time()
                if current_time - last_print_time >= 1:
                    print(print_statement)
                    last_print_time = current_time  # Update last print time
    
                if not at_waypoint:
                    if abs(pose[0] - WAYPOINTS[waypoint][0]) > 0.5 or abs(pose[1] - WAYPOINTS[waypoint][1]) > 0.5:
                        if initially_turning:
                            initial_theta = np.degrees(np.atan2(WAYPOINTS[waypoint][1] - pose[1], WAYPOINTS[waypoint][0] - pose[0]))
                            initial_theta -= 90  # Adjust for camera orientation
                            initial_theta = initial_theta % 360  # Normalize theta to be between 0 and 360 degrees
                            if abs(pose[2] - initial_theta) >= 5:
                                turn_to(initial_theta)
                            elif abs(pose[2] - initial_theta) < 5:
                                initially_turning = False
                                distance_to_obstacle = checkForObstacles("realsense-247122073398", frames)
                                if distance_to_obstacle is not None and distance_to_obstacle > 0.5:
                                    temp_waypoint_x = pose[0] + (distance_to_obstacle - 0.5) * np.sin(np.radians(pose[2]))
                                    temp_waypoint_y = pose[1] + (distance_to_obstacle - 0.5) * np.cos(np.radians(pose[2]))
                                    temp_waypoint = [temp_waypoint_x, temp_waypoint_y, "temporary"]
                                    WAYPOINTS.insert(waypoint, temp_waypoint)  # Insert the temporary waypoint before the current waypoint
                                    print(f"Obstacle detected, moving to temporary waypoint at position {temp_waypoint}")
                            
                        elif not initially_turning:
                            move_to(pose, WAYPOINTS[waypoint])
                            print_statement = f"Moving to waypoint {waypoint + 1} at position {pose}"
                    
                    else:
                        print(f"Arrived at waypoint {waypoint + 1} at position {pose}.")
                        stop_all()
                        at_waypoint = True  # Set flag to indicate arrival at waypoint
                
                else:
                    if WAYPOINTS[waypoint][2] == "mine":
                        print(f"Excavating at waypoint {waypoint + 1} at position {pose}.")
                        if i == 0:
                            initial_excavation_time = time.time()
                            i += 1
                            
                        if excavate(initial_excavation_time):
                            at_waypoint = False
                            waypoint += 1
                            i = 0
                    
                    elif WAYPOINTS[waypoint][2] == "deposit":
                        print(f"Depositing at waypoint {waypoint + 1} at position {pose}.")
                        if i == 0:
                            initial_deposit_time = time.time()
                            i += 1

                        if deposit(initial_deposit_time):
                            at_waypoint = False
                            waypoint += 1
                            i = 0
                    
                    elif WAYPOINTS[waypoint][2] == "temporary":
                        print("Arrived at temporary waypoint, avoiding obstacle.")
                        break


            if cv2.waitKey(1) == ord('q'):
                break

        # cv2.destroyAllWindows()
        stop_all()
        print(f"Final Pose: {pose}")
except Exception as e:
    print(f"An error occurred: {e}")
    stop_all()
finally:
     stop_all()