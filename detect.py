import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from dotenv import load_dotenv
from groq import Groq
from ultralytics import YOLO
from cobot import Cobot, Dirn

load_dotenv()

class CommandParser:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def parse(self, text):
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Extract target object noun. Output strictly object name in lowercase. No articles/verbs. If none, output 'none'."},
                    {"role": "user", "content": text}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0
            )
            return completion.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error parsing command: {e}")
            return "none"


class BBoxTracker:
    def __init__(self):
        self.last_center = None
        self.last_time = None
        self.velocity = (0.0, 0.0)
        self.is_predicting = False
    
    def update(self, center):
        now = time.time()
        if self.last_center is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                vx = (center[0] - self.last_center[0]) / dt
                vy = (center[1] - self.last_center[1]) / dt
                alpha = 0.3
                self.velocity = (
                    alpha * vx + (1 - alpha) * self.velocity[0],
                    alpha * vy + (1 - alpha) * self.velocity[1]
                )
        self.last_center = center
        self.last_time = now
        self.is_predicting = False
    
    def predict(self):
        if self.last_center is None or self.last_time is None:
            return None
        now = time.time()
        dt = now - self.last_time
        if dt > 2.0:
            return None
        predicted_x = self.last_center[0] + self.velocity[0] * dt
        predicted_y = self.last_center[1] + self.velocity[1] * dt
        self.is_predicting = True
        return (int(predicted_x), int(predicted_y))
    
    def reset(self):
        self.last_center = None
        self.last_time = None
        self.velocity = (0.0, 0.0)
        self.is_predicting = False


def intent_identification(parser):
    try:
        user_input = input("Enter query: ").strip()
        if not user_input:
            return None
    except (EOFError, KeyboardInterrupt):
        return None
    target_obj = parser.parse(user_input)
    if target_obj == 'none':
        print("Could not identify object.")
        return None
    return target_obj


def object_detection(model, img):
    results = model.predict(img, verbose=False, conf=0.25)
    if results and len(results) > 0:
        print(f"Number of detections: {len(results[0].boxes)}")
        r = results[0]
        if len(r.boxes) > 0:
            return r.boxes[0]
    return None


def get_bbox_center(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    ox = int((x1 + x2) / 2)
    oy = int((y1 + y2) / 2)
    return (ox, oy), (x1, y1, x2, y2)


def pickup(bot):
    print("Executing Pickup Sequence...")
    bot.setVelocity(2)
    bot.stopJogging()
    bot.jogCartesianRelative(Dirn.NEGATIVE, 0)
    time.sleep(3)
    bot.stopJogging()
    bot.jogCartesianRelative(Dirn.POSITIVE, 1)
    time.sleep(2.5)
    bot.jogCartesianRelative(Dirn.NEGATIVE, 2)
    print("Going towards the object...")
    time.sleep(10)
    bot.stopJogging()
    bot.jogCartesianRelative(Dirn.NEGATIVE, 2)
    time.sleep(5)



def shutdown(bot, pipeline):
    print("Shutting down...")
    bot.stopJogging()
    time.sleep(0.5)
    bot.baseRigid()
    print("Returned to base position sucessfully.")
    time.sleep(2)
    pipeline.stop()
    cv2.destroyAllWindows()
    bot.disconnect()


def detect():
    bot = Cobot("10.202.4.214", "cobot1234")
    pipeline = None
    
    try:
        bot.connect()
        bot.setVelocity(2)
        print("Connected to Cobot successfully.")
    except Exception as e:
        print(f"Failed to connect to Cobot: {e}")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start Realsense pipeline: {e}")
        bot.disconnect()
        return

    align = rs.align(rs.stream.color)
    model = YOLO("yoloe-26l-seg.pt")
    parser = CommandParser()
    tracker = BBoxTracker()
    
    inference_interval = 0.1
    last_inference_time = 0
    threshold = 40
    
    try:
        target_obj = intent_identification(parser)
        if target_obj is None:
            print("No valid target. Exiting.")
            shutdown(bot, pipeline)
            return
            
        print(f"Tracking: {target_obj}")
        
        try:
            model.set_classes([target_obj])
        except Exception:
            pass
        
        detected_once = False
        x_aligned = False
        y_aligned = False
        last_box_coords = None
        current_command_x = "SEARCHING"
        current_command_y = ""

        print("Searching for object...")
        bot.jogJoint(Dirn.NEGATIVE, 5)
               
        while True:
            frames = pipeline.poll_for_frames()
            if not frames:
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            h, w, _ = img.shape
            cx, cy = w // 2, h // 2

            now = time.time()
            center = None
            using_prediction = False
            
            if now - last_inference_time > inference_interval:
                last_inference_time = now
                
                box = object_detection(model, img)
                
                if box is not None:
                    center, box_coords = get_bbox_center(box)
                    tracker.update(center)
                    last_box_coords = box_coords
                    using_prediction = False
                    
                    if not detected_once:
                        detected_once = True
                        print("Object detected, stopping search...")
                        bot.stopJogging()
                        
                else:
                    if detected_once:
                        predicted = tracker.predict()
                        if predicted is not None:
                            center = predicted
                            using_prediction = True
                            print(f"Using predicted center: {center}")
                        else:
                            print("Lost object, prediction unavailable. Stopping.")
                            bot.stopJogging()
                            
                            bot.jogJoint(Dirn.NEGATIVE, 5)
                    else:
                        current_command_x = "SEARCHING"
                        current_command_y = ""
                
                if center is not None:
                    ox, oy = center
                    dx = ox - cx 
                    dy = oy - cy 
                    
                    status = "[PREDICTED]" if using_prediction else ""
                    print(f"{status} Center: ({ox}, {oy}) | Deviation: (dx: {dx}, dy: {dy})")
                    
                    x_aligned = abs(dx) < threshold
                    y_aligned = abs(dy) < threshold
                    
                    if x_aligned and y_aligned:
                        print("Both axes aligned. Initiating Pickup.")
                        current_command_x = "STOP"
                        current_command_y = "STOP"
                        
                        ox_safe = min(max(ox, 0), w - 1)
                        oy_safe = min(max(oy, 0), h - 1)
                        dist = depth_frame.get_distance(ox_safe, oy_safe)
                        print(f"Distance to Object: {dist:.3f} meters")
                        camera_height_m = 0.54
                        if dist >= 0:
                            object_height = camera_height_m - dist
                            print(f"Final Object Height: {object_height:.3f} meters")
                        pickup(bot)
                        
                        
                        break
                    
                    else:
                        bot.setVelocity(3)
                   
                        time.sleep(0.02)
                        
                        if x_aligned:
                            current_command_x = "ALIGNED_X"
                        elif dx > threshold:
                            current_command_x = "RIGHT"
                            bot.jogCartesianRelative(Dirn.POSITIVE, 0)
                        elif dx < -threshold:
                            current_command_x = "LEFT"
                            bot.jogCartesianRelative(Dirn.NEGATIVE, 0)
                        
                        if y_aligned:
                            current_command_y = "ALIGNED_Y"
                        elif dy > threshold:
                            current_command_y = "BACKWARD"
                            bot.jogCartesianRelative(Dirn.NEGATIVE, 1)
                        elif dy < -threshold:
                            current_command_y = "FORWARD"
                            bot.jogCartesianRelative(Dirn.POSITIVE, 1)

            if last_box_coords:
                x1, y1, x2, y2 = last_box_coords
                color = (0, 255, 255) if using_prediction else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if center:
                    cv2.circle(img, center, 5, (0, 0, 255), -1)

            cv2.line(img, (cx, 0), (cx, h), (200, 200, 200), 1)
            cv2.line(img, (0, cy), (w, cy), (200, 200, 200), 1)
            
            overlay_text = f"{current_command_x} | {current_command_y}"
            cv2.putText(img, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Tracker", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        shutdown(bot, pipeline)


if __name__ == "__main__":
    detect()