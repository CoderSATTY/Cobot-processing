import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
try:
    pipeline.start(config)
except Exception:
    sys.exit(0)

while True:
    frames = pipeline.poll_for_frames()
    if not frames:
        continue
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())

    cv2.imshow('Color Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()