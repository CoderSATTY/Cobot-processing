import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def get_coordinates_in_initial_frame(
    d, px, py, depth_intrin,
    yaw_rad, cam_height_cm, tilt_deg
):
    tilt_rad = np.radians(tilt_deg)

    # Deproject
    x_c, y_c, z_c = rs.rs2_deproject_pixel_to_point(
        depth_intrin, [px, py], d
    )

    # meters → cm
    x_c *= 100
    y_c *= 100
    z_c *= 100

    #Pitch (tilt about X) 
    x1 = x_c
    y1 = z_c * np.cos(tilt_rad) 
    z1 = z_c * np.sin(tilt_rad)

    #z_w =  z1

    # Yaw (about Z)
    #x_w = x1 * np.cos(yaw_rad) - y1 * np.sin(yaw_rad)
    #y_w = x1 * np.sin(yaw_rad) + y1 * np.cos(yaw_rad)

    #return x_w, y_w, z_w
    return x1, y1, z1


try:
    while True:
        CAMERA_HEIGHT_CM = 52.0  
        CAMERA_TILT_DEG = 60.0     
        CURRENT_YAW_DEG = 0  
        
        yaw_rad = np.radians(CURRENT_YAW_DEG)
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        bbox_i = [198.82777404785156, 0.29718017578125, 415.8533935546875, 253.65660095214844]  
        bbox = [int(b) for b in bbox_i]
        px, py = (bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2

        width, height = depth_intrin.width, depth_intrin.height
        ocx, ocy = width//2, height//2
        #px, py = width // 2, height // 2
        
        dist = depth_frame.get_distance(px, py)
        
        if dist > 0:
            rx, ry, rz = get_coordinates_in_initial_frame(
                dist, px, py, depth_intrin, yaw_rad, CAMERA_HEIGHT_CM, CAMERA_TILT_DEG
            )

            print(f"Initial Frame -> X: {rx:6.2f} | Y: {ry:6.2f} | Z (Height): {rz:6.2f} cm")

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        cv2.circle(color_image, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(color_image, (ocx, ocy), 5, (255, 0, 0), -1)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        
        # Stack images safely
        if depth_colormap.shape != color_image.shape:
            color_image = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
            
        cv2.imshow('RealSense', np.hstack((color_image, depth_colormap)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()