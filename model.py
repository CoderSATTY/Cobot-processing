import os
import subprocess
import torch
import cv2
import numpy as np
import supervision as sv
from typing import List

subprocess.run(["nvidia-smi"])

HOME = os.getcwd()
print("HOME:", HOME)

subprocess.run(["which", "pip"])
subprocess.run(["pip", "--version"])
subprocess.run(["python", "-m", "ensurepip", "--upgrade"])
subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
subprocess.run([
    "python", "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

os.chdir("GroundingDINO")
subprocess.run(["python", "-m", "pip", "install", "-e", ".", "--no-build-isolation"])
os.chdir("..")

os.makedirs("weights", exist_ok=True)

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    HOME, "weights", "groundingdino_swint_ogc.pth"
)

if not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
    subprocess.run([
        "wget",
        "-O", "weights/groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ])

print(GROUNDING_DINO_CHECKPOINT_PATH, os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

GROUNDING_DINO_CONFIG_PATH = os.path.join(
    HOME,
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinT_OGC.py"
)

print(GROUNDING_DINO_CONFIG_PATH, os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

from groundingdino.util.inference import Model

grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    device=str(DEVICE)
)
grounding_dino_model.model = grounding_dino_model.model.to("cuda")

subprocess.run([
    "python", "-m", "pip", "install",
    "git+https://github.com/facebookresearch/segment-anything.git"
])

subprocess.run(["pip", "uninstall", "-y", "supervision"])
subprocess.run(["pip", "install", "supervision==0.27.0"])

import supervision as sv

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

if not os.path.exists(SAM_CHECKPOINT_PATH):
    subprocess.run([
        "wget",
        "-q",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "-O", SAM_CHECKPOINT_PATH
    ])

print(SAM_CHECKPOINT_PATH, "; exists:", os.path.isfile(SAM_CHECKPOINT_PATH))

from segment_anything import sam_model_registry, SamPredictor

SAM_ENCODER_VERSION = "vit_h"

sam = sam_model_registry[SAM_ENCODER_VERSION](
    checkpoint=SAM_CHECKPOINT_PATH
).to(device=DEVICE)

sam_predictor = SamPredictor(sam)

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]

SOURCE_IMAGE_PATH = "image1.jpeg"
CLASSES = ["bottle", "book"]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image = cv2.imread(SOURCE_IMAGE_PATH)

detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=enhance_class_name(class_names=CLASSES),
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

box_annotator = sv.BoxAnnotator()

annotated_frame = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

cv2.imwrite("detected.png", annotated_frame)

def segment(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    masks_out = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        masks_out.append(masks[np.argmax(scores)])
    return np.array(masks_out)

detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

segmented_image = mask_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

segmented_image = box_annotator.annotate(
    scene=segmented_image,
    detections=detections
)

cv2.imwrite("segmented.png", segmented_image)
