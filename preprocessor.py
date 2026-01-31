import cv2
import time
import torch
import os
from ultralytics import YOLO
from groq import Groq

# Configuration
CAMERA_INDEX = 1
MODEL_PATH = "weights/yoloe-26l-seg.pt"
SAVE_DIR = "images"
SYSTEM_PROMPT_PATH = "system_prompt.txt"

CONF_THRES = 0.05
EDGE_MARGIN = 15        # Pixel buffer from edge (excludes partial objects)
REQUIRED_STABILITY = 4  # Number of consistent frames required before capturing

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def load_system_prompt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_object_intent(user_input, system_prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=50,
        )
        
        object_name = chat_completion.choices[0].message.content.strip()
        return object_name
    
    except Exception as e:
        print(f"Error with Groq API: {e}")
        print("Falling back to direct input...")
        return user_input.strip()


def is_box_fully_in_frame(box, frame_shape, margin):
    """
    Returns True if the bounding box is completely inside the frame 
    (not touching edges within the margin).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    
    # Check bounds
    if x1 < margin: return False         # Touching Left
    if y1 < margin: return False         # Touching Top
    if x2 > (w - margin): return False   # Touching Right
    if y2 > (h - margin): return False   # Touching Bottom
    
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    
    # Load YOLO model
    start = time.perf_counter()
    model = YOLO(MODEL_PATH).to(device)
    end = time.perf_counter()
    print(f"Model loading time: {end - start:.2f}s")

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    print("\nObject Detection System Ready!")

    # QUERY LOOP
    while True:
        try:
            user_input = input("\nWhat would you like to detect? (or 'q' to quit): ").strip()
        except EOFError:
            break
            
        if not user_input or user_input.lower() == 'q':
            break
        
        # Extract object intent using LLM
        print(f"\nProcessing: '{user_input}'")
        target_class = extract_object_intent(user_input, system_prompt)
        print(f"Detected object: '{target_class}'")
        
        model.set_classes([target_class])
        print(f"Starting detection for: {target_class}")
        print("-" * 50)

        detected = False
        stability_counter = 0  

        while not detected:
            start_loop = time.time()
            ret, frame = cap.read()
            
            if not ret:
                continue

            results = model.predict(
                frame,
                conf=CONF_THRES,
                task="segment",
                verbose=False
            )[0]

            # Filter Detections
            valid_box = None
            
            if results.boxes is not None and len(results.boxes) > 0:
                # Find the best box that is fully in frame
                for i, box in enumerate(results.boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    if is_box_fully_in_frame(coords, frame.shape, EDGE_MARGIN):
                        valid_box = results[i]  # This is a valid complete object
                        final_coords = coords  
                        break 

            # Stability Logic
            annotated = frame.copy()
            
            if valid_box:
                stability_counter += 1
                annotated = valid_box.plot()
                
                # Display stability status
                status_text = f"Stable: {stability_counter}/{REQUIRED_STABILITY}"
                cv2.putText(annotated, status_text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if stability_counter >= REQUIRED_STABILITY:
                    detected = True  # Trigger exit
                
            else:
                # Object lost or touching edge -> Reset counter
                if stability_counter > 0:
                    print(f"Stability lost (count was {stability_counter})")
                stability_counter = 0
                cv2.putText(annotated, "Searching...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow("Live Feed", annotated)
            
            # Save Logic (Only if detected flag is set by stability logic)
            if detected:
                timestamp = int(time.time())
                filename = f"{SAVE_DIR}/{target_class.replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\n✓ Saved detection: {filename}")
                print(f"  Coordinates: {final_coords.tolist()}")
                cv2.waitKey(500) 

            latency_ms = (time.time() - start_loop) * 1000
            cv2.setWindowTitle(
                "Live Feed",
                f"Live Feed | Latency: {latency_ms:.1f} ms"
            )
            
            # Exit CURRENT inference only
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping current inference.")
                break

        # Destroy windows after each query 
        cv2.destroyAllWindows()

        # Clear CUDA cache after each query 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nShutting down...")


if __name__ == "__main__":
    main()