# Cobot Processing System: Autonomous Object Retrieval

This project implements an autonomous object tracking and retrieval system using a **Syncro 5 Addverb Cobot**, **Realsense Depth Camera**, and **LLMs** for intent recognition.

It features a custom **TCP-based Cobot Controller SDK** that bypasses ROS for lightweight, direct control.

## 🚀 Key Features
- **Natural Language Command Interface**: Uses **Groq API (Llama 3)** to interpret commands (e.g., "pick up the red bottle").
- **Custom Cobot SDK**: Direct TCP/IP control of the robot without ROS middleware.
- **Real-time Visual Servoing**: Closed-loop control using YOLO detection to align the robot arm.
- **Depth Sensing**: 3D localization using Intel RealSense.

## 🛠️ Tech Stack

### Hardware
*   **Robot**: Syncro 5 (Addverb)
*   **Camera**: Intel RealSense D435i

### Software
*   **Python 3.10+**
*   **Cobot Controller SDK**: Custom Python wrapper for TCP socket communication.
*   **Ultralytics YOLOv8**: Object Detection.
*   **Groq API**: LLM Inference.
*   **pyrealsense2**: Depth processing.

---

## 🤖 Cobot Controller SDK (@cobot)

The system uses a custom python client library (made by teammate) to communicate with a C++ control server running on the robot hardware.

**Protocol:** TCP/IP Sockets (Port 5000)
**Architecture:**
*   **Server (C++)**: Runs on the robot, manages hardware safety, and executing motion primitives.
*   **Client (Python)**: Sends byte-code commands (e.g., `b"j+1"` for "Jog Joint 1 Positive").

### Key Methods
The `Cobot` class abstracts the socket handling:

```python
from cobot import Cobot, Dirn

with Cobot("192.168.x.x", "password") as bot:
    bot.connect()
    
    # Movement Control
    bot.setVelocity(2.0)                  # Set speed scalar
    bot.jogJoint(Dirn.POSITIVE, 0)        # Move Base Joint
    bot.jogCartesianRelative(Dirn.NEGATIVE, 1) # Move Tool Frame Y-axis
    bot.stopJogging()                     # Immediate Halt
    
    # Gipper open & close
    bot.gripperOpen()
    bot.gripperClose()
    
    # Recovery
    bot.baseRigid()                       # Return to home position
```

---

## 🔄 Autonomous Pipeline (`detect.py`)

The `detect.py` script orchestrates the full autonomous retrieval loop. Here is the step-by-step pipeline:

### 1. Initialization & Connection
*   Connects to the Cobot via TCP (`bot.connect()`).
*   Starts the RealSense pipeline (Color + Depth streams).
*   Loads the YOLO model and Groq client.

### 2. Intent Recognition
*   **Input**: User types "Find the blue cup".
*   **LLM Processing**: `CommandParser` uses Llama 3 via Groq to exact the target class -> `cup`.
*   **YOLO Config**: The model resets to track only the `cup` class.

### 3. Search Phase
*   The robot executes a `jogJoint` command to scan the environment until the target object is detected by YOLO.

### 4. Visual Servoing (Alignment Loop)
Once detected, the system enters a PID-like control loop to center the object in the camera frame:

*   **Error Calculation**:
    *   `dx = object_center_x - frame_center_x`
    *   `dy = object_center_y - frame_center_y`
*   **Control Logic**:
    *   **X-Axis Alignment**:
        *   If `dx > threshold`: Command `jogCartesianRelative(POSITIVE, 0)` (Move Right).
        *   If `dx < -threshold`: Command `jogCartesianRelative(NEGATIVE, 0)` (Move Left).
    *   **Y-Axis Alignment**:
        *   If `dy > threshold`: Command `jogCartesianRelative(NEGATIVE, 1)` (Move Back).
        *   If `dy < -threshold`: Command `jogCartesianRelative(POSITIVE, 1)` (Move Forward).
    *   **Distance Check**: The Depth camera verifies the object is within reach.

### 5. Pickup Sequence
When `dx` and `dy` are both within the threshold (Aligned):
1.  **Stop**: `bot.stopJogging()`
2.  **Approach**: Move down (`jogCartesianRelative`) and forward to the object coordinates.
3.  **Grasp**: `bot.gripperClose()` (Logic reserved but not fully implemented in provided script).
4.  **Retract**: `bot.baseRigid()` to lift the object.

---

## 📂 File Structure

```
.
├── detect.py            # Main autonomous servoing loop
├── preprocessor.py      # Camera calibration utility
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```