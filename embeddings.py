from typing import List, Dict, Any

import numpy as np
from PIL import Image

from models import run_arcface
import onnxruntime as ort

def preprocess_face_crops_for_arcface(face_dicts: List[Dict[str, Any]], size: int = 112) -> np.ndarray:
    """
    Resize and normalize face crops to ArcFace NHWC input format.
    Returns: (N, size, size, 3) float32.
    """
    processed = []
    for f in face_dicts:
        crop = f["crop"]
        img = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        # Keep HWC order – do NOT transpose to CHW
        processed.append(arr)
    if not processed:
        return np.empty((0, size, size, 3), dtype="float32")
    batch = np.stack(processed, axis=0)
    return batch

def compute_face_embeddings(
    arcface_session: ort.InferenceSession,
    face_dicts: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Given face crops, return L2-normalized embeddings (N, D).
    """
    batch = preprocess_face_crops_for_arcface(face_dicts)
    if batch.shape[0] == 0:
        return np.empty((0, 512), dtype="float32")
    raw_embeddings = run_arcface(arcface_session, batch)
    # L2-normalize.
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-9
    normalized = raw_embeddings / norms
    return normalized.astype("float32")
