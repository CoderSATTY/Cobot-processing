"""
Video Feed Object Detection & Segmentation Pipeline
Uses GroundingDINO + SAM with NLP query parsing for on-demand processing.

Usage:
    python video_pipeline.py

Type natural language queries (e.g., "pick up the orange") and press Enter.
Type 'quit' to exit.
"""

import os
import cv2
import torch
import numpy as np
import supervision as sv
from typing import List, Tuple, Optional
from groq import Groq  # Change to 'from openai import OpenAI' if using OpenAI

# ============================================================================
# Configuration
# ============================================================================

HOME = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    HOME, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

# Detection thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Camera settings
CAMERA_INDEX = 0  # Change if using external camera

# LLM API key (set your key here or via environment variable)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-api-key-here")

# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    """Load GroundingDINO and SAM models."""
    print(f"Loading models on {DEVICE}...")
    
    # Load GroundingDINO
    from groundingdino.util.inference import Model
    grounding_dino = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=str(DEVICE)
    )
    grounding_dino.model = grounding_dino.model.to(DEVICE)
    
    # Load SAM
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    print("Models loaded successfully!")
    return grounding_dino, sam_predictor

# ============================================================================
# NLP Query Parser
# ============================================================================

def parse_query_to_objects(query: str) -> List[str]:
    """
    Parse natural language query to extract object names.
    
    Example:
        "pick up the orange" -> ["orange"]
        "find all fruits on the table" -> ["apple", "orange", "banana", "table"]
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an object extraction assistant for a robotic vision system.
Given a natural language command, extract the specific object(s) that need to be detected in an image.

Rules:
- Return ONLY a comma-separated list of simple object names
- Use singular nouns (e.g., "orange" not "oranges")
- Keep names simple and concrete (e.g., "cup" not "drinking vessel")
- If the query mentions an action (pick, grab, find), focus on the TARGET object
- Do not include abstract concepts, only physical objects

Examples:
- "pick up the orange" -> orange
- "grab the red cup near the laptop" -> cup, laptop
- "find all electronics" -> phone, laptop, tablet, keyboard, mouse
- "move the book to the shelf" -> book, shelf
"""
            },
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        max_tokens=100
    )
    
    objects_str = response.choices[0].message.content.strip()
    objects = [obj.strip().lower() for obj in objects_str.split(",") if obj.strip()]
    
    print(f"Parsed objects: {objects}")
    return objects

# ============================================================================
# Detection & Segmentation
# ============================================================================

def enhance_class_name(class_names: List[str]) -> List[str]:
    """Enhance class names for better detection."""
    return [f"all {name}s" for name in class_names]

def segment_objects(sam_predictor, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Generate segmentation masks for detected boxes."""
    sam_predictor.set_image(image)
    masks = []
    for box in boxes:
        mask_predictions, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        masks.append(mask_predictions[np.argmax(scores)])
    return np.array(masks) if masks else np.array([])

def process_frame(
    frame: np.ndarray,
    objects: List[str],
    grounding_dino,
    sam_predictor
) -> Tuple[np.ndarray, dict]:
    """
    Process a single frame with detection and segmentation.
    
    Returns:
        annotated_image: Image with boxes and masks drawn
        results: Dictionary with detection details
    """
    # Detect objects
    detections = grounding_dino.predict_with_classes(
        image=frame,
        classes=enhance_class_name(objects),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    results = {
        "objects_queried": objects,
        "num_detections": len(detections.xyxy),
        "boxes": detections.xyxy.tolist() if len(detections.xyxy) > 0 else [],
        "confidences": detections.confidence.tolist() if len(detections.confidence) > 0 else [],
        "class_ids": detections.class_id.tolist() if detections.class_id is not None else []
    }
    
    # Segment if detections found
    if len(detections.xyxy) > 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections.mask = segment_objects(sam_predictor, frame_rgb, detections.xyxy)
        
        # Annotate image
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
    else:
        annotated = frame.copy()
        print("No objects detected!")
    
    return annotated, results

# ============================================================================
# Video Capture Handler
# ============================================================================

class VideoCapture:
    """Wrapper for video capture with frame grabbing on demand."""
    
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        print(f"Camera {camera_index} opened successfully")
    
    def grab_frame(self) -> Optional[np.ndarray]:
        """Capture current frame from video feed."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame
    
    def show_preview(self, window_name: str = "Preview"):
        """Show live preview (non-blocking)."""
        frame = self.grab_frame()
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
    
    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main interactive loop."""
    print("\n" + "="*60)
    print("Video Feed Object Detection & Segmentation Pipeline")
    print("="*60 + "\n")
    
    # Load models
    grounding_dino, sam_predictor = load_models()
    
    # Initialize camera
    video = VideoCapture(CAMERA_INDEX)
    
    print("\nReady! Type a command (e.g., 'pick up the orange') or 'quit' to exit.")
    print("Press 'p' in the preview window to show live feed.\n")
    
    try:
        while True:
            # Show live preview
            video.show_preview("Live Feed - Press 'q' to close preview")
            
            # Check for keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Live Feed - Press 'q' to close preview")
            
            # Get user query (with timeout to allow preview updates)
            import select
            import sys
            
            # Simple input for Windows compatibility
            print("\nEnter query (or 'quit'): ", end="", flush=True)
            query = input().strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            # Process query
            print(f"\nProcessing: '{query}'")
            
            # 1. Parse query to objects
            objects = parse_query_to_objects(query)
            
            if not objects:
                print("Could not extract objects from query. Try again.")
                continue
            
            # 2. Capture frame
            print("Capturing frame...")
            frame = video.grab_frame()
            
            if frame is None:
                print("Failed to capture frame!")
                continue
            
            # 3. Run detection + segmentation pipeline
            print("Running detection & segmentation...")
            annotated_frame, results = process_frame(
                frame, objects, grounding_dino, sam_predictor
            )
            
            # 4. Display and save results
            print(f"\nResults: {results['num_detections']} object(s) detected")
            for i, (box, conf) in enumerate(zip(results['boxes'], results['confidences'])):
                print(f"  [{i+1}] Box: {[int(x) for x in box]}, Confidence: {conf:.2f}")
            
            # Show result
            cv2.imshow("Detection Result", annotated_frame)
            
            # Save result
            output_path = f"result_{len(os.listdir('.'))}.png"
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved to: {output_path}")
            
            print("\nPress any key in the result window to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Detection Result")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        video.release()
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()
