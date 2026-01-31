# Object Detection System with LLM Intent Recognition

An intelligent object detection system that uses Groq LLM for natural language intent extraction and YOLO for real-time object detection.

## Features

- **Natural Language Input**: Describe what you want to detect in plain English
- **LLM Intent Extraction**: Groq-powered agent extracts the object name from your query
- **Real-time Detection**: YOLO-based object detection with live camera feed
- **Stability Checking**: Ensures objects are fully in frame before capturing
- **Automatic Saving**: Saves detected objects with timestamps

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Groq API Key**:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   
   Or on Windows:
   ```cmd
   set GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Prepare Model**:
   - Place your YOLO model at `weights/yoloe-26l-seg.pt`
   - Or update `MODEL_PATH` in the script

4. **Configure Camera**:
   - Update `CAMERA_INDEX` if needed (default is 1)

## Usage

Run the script:
```bash
python preprocessor.py
```

Example interactions:
```
What would you like to detect? Find my water bottle
Processing: 'Find my water bottle'
Detected object: 'bottle'
Starting detection for: bottle

What would you like to detect? I need to see if there's a person in the room
Processing: 'I need to see if there's a person in the room'
Detected object: 'person'
Starting detection for: person

What would you like to detect? Detect my laptop
Processing: 'Detect my laptop'
Detected object: 'laptop'
Starting detection for: laptop
```

## How It Works

1. **User Input**: Enter a natural language query describing what to detect
2. **LLM Processing**: Groq LLM extracts the specific object name
3. **YOLO Detection**: YOLO model detects the object in real-time
4. **Stability Check**: System waits for object to be stable and fully in frame
5. **Auto-Save**: Image is saved when detection is stable

## Configuration

Edit these parameters in `preprocessor.py`:

```python
CAMERA_INDEX = 1              # Camera to use
MODEL_PATH = "weights/..."    # YOLO model path
CONF_THRES = 0.05            # Detection confidence threshold
EDGE_MARGIN = 15             # Pixel buffer from edges
REQUIRED_STABILITY = 4       # Frames needed for stable detection
```

## System Prompt

The LLM behavior is controlled by `system_prompt.txt`. This file contains:
- Instructions for object extraction
- Few-shot examples for better accuracy
- Rules for handling different input formats

You can modify this file to improve intent recognition for your specific use case.

## Troubleshooting

**Camera not opening**: 
- Check `CAMERA_INDEX` value
- Try different values (0, 1, 2, etc.)

**Groq API errors**:
- Verify your API key is set correctly
- Check internet connection
- System will fall back to direct input if API fails

**Detection issues**:
- Adjust `CONF_THRES` for sensitivity
- Modify `EDGE_MARGIN` for edge detection
- Change `REQUIRED_STABILITY` for faster/slower capture

## File Structure

```
.
├── preprocessor.py          # Main script
├── system_prompt.txt        # LLM system prompt with examples
├── requirements.txt         # Python dependencies
├── weights/
│   └── yoloe-26l-seg.pt    # YOLO model
└── images/                  # Saved detections (auto-created)
```

## License

This project uses:
- Ultralytics YOLO (AGPL-3.0)
- Groq API (check Groq's terms)
- OpenCV (Apache 2.0)