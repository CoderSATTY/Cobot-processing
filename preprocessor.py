import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from typing import List, Tuple, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
HOME = os.getcwd()
sys.path.insert(0, os.path.join(HOME, "GroundingDINO"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_CONFIG_PATH = os.path.join(
    HOME, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

CAMERA_INDEX = 1

GROQ_API_KEY = os.getenv("GROQ_KEY")


def load_models():
    print(f"Loading models on {DEVICE}...")
    
    from groundingdino.util.inference import Model
    grounding_dino = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=str(DEVICE)
    )
    grounding_dino.model = grounding_dino.model.to(DEVICE)
    
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    print("Models loaded successfully!")
    return grounding_dino, sam_predictor


def parse_query_to_objects(query: str) -> List[str]:
    client = Groq(api_key=GROQ_API_KEY)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an object extraction assistant for a robotic vision system.
Given a natural language command, extract ONLY the TARGET object on which the action is to be performed.

Rules:
- Return ONLY the single target object name that the action is being performed ON
- IGNORE reference objects used for location (e.g., "near cup", "beside table", "on shelf")
- Use singular nouns (e.g., "button" not "buttons")
- Keep names simple and concrete
- Focus ONLY on what is being acted upon - pick, grab, press, move, find, etc.

Examples:
- "pick up the orange" -> orange
- "grab the red cup near the laptop" -> cup
- "press the emergency stop button near cup" -> stop button
- "move the book to the shelf" -> book
- "find the screwdriver next to the toolbox" -> screwdriver
- "press the power button on the machine" -> power button
"""
            },
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    objects_str = response.choices[0].message.content.strip()
    target_object = objects_str.split(",")[0].strip().lower() if objects_str else ""
    
    print(f"Target object: {target_object}")
    return target_object


def enhance_class_name(class_name: str) -> List[str]:
    return [f"all {class_name}s"]


def segment_objects(sam_predictor, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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
    target_object: str,
    grounding_dino,
    sam_predictor
) -> Tuple[np.ndarray, dict]:
    detections = grounding_dino.predict_with_classes(
        image=frame,
        classes=enhance_class_name(target_object),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    results = {
        "target_object": target_object,
        "num_detections": len(detections.xyxy),
        "boxes": detections.xyxy.tolist() if len(detections.xyxy) > 0 else [],
        "confidences": detections.confidence.tolist() if len(detections.confidence) > 0 else [],
        "class_ids": detections.class_id.tolist() if detections.class_id is not None else []
    }
    
    if len(detections.xyxy) > 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections.mask = segment_objects(sam_predictor, frame_rgb, detections.xyxy)
        
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
    else:
        annotated = frame.copy()
        print(f"No '{target_object}' detected!")
    
    return annotated, results


class VideoCapture:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        print(f"Camera {camera_index} opened successfully")
    
    def grab_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame
    
    def show_preview(self, window_name: str = "Preview"):
        frame = self.grab_frame()
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("Video Feed Object Detection & Segmentation Pipeline")
    print("="*60 + "\n")
    
    grounding_dino, sam_predictor = load_models()
    
    video = VideoCapture(CAMERA_INDEX)
    
    print("\nReady! Type a command (e.g., 'pick up the orange') or 'quit' to exit.")
    print("Press 'p' in the preview window to show live feed.\n")
    
    try:
        while True:
            video.show_preview("Live Feed - Press 'q' to close preview")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Live Feed - Press 'q' to close preview")
            
            import select
            import sys
            
            print("\nEnter query (or 'quit'): ", end="", flush=True)
            query = input().strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: '{query}'")
            
            target_object = parse_query_to_objects(query)
            
            if not target_object:
                print("Could not extract target object from query. Try again.")
                continue
            
            print("Capturing frame...")
            frame = video.grab_frame()
            
            if frame is None:
                print("Failed to capture frame!")
                continue
            
            print(f"Running detection & segmentation for '{target_object}'...")
            annotated_frame, results = process_frame(
                frame, target_object, grounding_dino, sam_predictor
            )
            
            print(f"\nResults: {results['num_detections']} '{target_object}' detected")
            for i, (box, conf) in enumerate(zip(results['boxes'], results['confidences'])):
                print(f"  [{i+1}] Box: {[int(x) for x in box]}, Confidence: {conf:.2f}")
            
            cv2.imshow("Detection Result", annotated_frame)
            
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
