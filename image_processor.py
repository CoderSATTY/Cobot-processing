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

GROQ_API_KEY = os.getenv("GROQ_KEY")

IMAGE_PATH = "image.png"  


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


def parse_query_to_target(query: str) -> str:
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
    
    target_object = response.choices[0].message.content.strip().lower()
    print(f"Target object: {target_object}")
    return target_object


def detect_and_segment(
    image: np.ndarray,
    target_object: str,
    grounding_dino,
    sam_predictor
) -> Tuple[np.ndarray, dict]:
    detections = grounding_dino.predict_with_classes(
        image=image,
        classes=[target_object],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    results = {
        "target_object": target_object,
        "num_detections": len(detections.xyxy),
        "boxes": detections.xyxy.tolist() if len(detections.xyxy) > 0 else [],
        "confidences": detections.confidence.tolist() if len(detections.confidence) > 0 else [],
    }
    
    if len(detections.xyxy) > 0:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_rgb)
        
        masks = []
        for box in detections.xyxy:
            mask_preds, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
            masks.append(mask_preds[np.argmax(scores)])
        detections.mask = np.array(masks)
        
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
    else:
        annotated = image.copy()
        print(f"No '{target_object}' detected!")
    
    return annotated, results


def load_image(image_path: str) -> Optional[np.ndarray]:
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        print(f"Loaded image: {image_path}")
        return image
    print(f"Image not found: {image_path}")
    return None


def main():
    print("\n" + "="*60)
    print("Object Detection & Segmentation Pipeline")
    print("="*60 + "\n")
    
    grounding_dino, sam_predictor = load_models()
    
    image = load_image(IMAGE_PATH)
    if image is None:
        print(f"Please set a valid IMAGE_PATH at the top of the file.")
        return
    
    print(f"\nImage loaded: {IMAGE_PATH}")
    print("Enter your query (e.g., 'pick up the orange') or 'quit' to exit.\n")
    
    try:
        while True:
            query = input("Query: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: '{query}'")
            
            target_object = parse_query_to_target(query)
            
            if not target_object:
                print("Could not extract target object. Try again.")
                continue
            
            print(f"Detecting '{target_object}'...")
            annotated_image, results = detect_and_segment(
                image, target_object, grounding_dino, sam_predictor
            )
            
            print(f"Results: {results['num_detections']} '{target_object}' found")
            for i, (box, conf) in enumerate(zip(results['boxes'], results['confidences'])):
                print(f"  [{i+1}] Box: {[int(x) for x in box]}, Confidence: {conf:.2f}")
            
            output_path = f"result_{target_object.replace(' ', '_')}.png"
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved: {output_path}\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    print("Done!")


if __name__ == "__main__":
    main()
