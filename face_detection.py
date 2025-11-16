from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import YOLO_CONFIDENCE_THRESHOLD

def list_image_paths(root: Path) -> List[Path]:
    """
    Recursively list common image files under root.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]

def load_image_rgb(path: Path) -> np.ndarray:
    """
    Load image as RGB numpy array (H, W, 3), uint8.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)

def detect_faces_in_image(yolo_model: YOLO, image_path: Path) -> List[Dict[str, Any]]:
    """
    Run YOLO face detection and return list of dicts with
    bounding box and cropped face image.
    """
    results = yolo_model(str(image_path), conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    if not results:
        return []

    img_np = load_image_rgb(image_path)
    h, w, _ = img_np.shape
    faces: List[Dict[str, Any]] = []

    for result in results:
        # boxes.xyxy is (N, 4) with [x1, y1, x2, y2].
        if result.boxes is None:
            continue
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_np[y1:y2, x1:x2, :]
            faces.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "crop": crop,
                }
            )
    return faces


def detect_faces_in_images_batch(
    yolo_model: YOLO, image_paths: List[Path]
) -> List[List[Dict[str, Any]]]:
    """
    Run YOLO face detection on multiple images at once (batch processing).
    Returns a list of face dicts per image (one list per input image).
    """
    if not image_paths:
        return []

    # Convert Path objects to strings for YOLO
    image_paths_str = [str(p) for p in image_paths]
    
    # Run batch inference
    results = yolo_model(image_paths_str, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    
    all_faces = []
    for img_path, result in zip(image_paths, results):
        img_np = load_image_rgb(img_path)
        h, w, _ = img_np.shape
        faces: List[Dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img_np[y1:y2, x1:x2, :]
                faces.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "crop": crop,
                    }
                )
        all_faces.append(faces)
    
    return all_faces