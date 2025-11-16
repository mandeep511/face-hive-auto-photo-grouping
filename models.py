from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np

from config import (
    FACE_DETECTOR_REPO,
    FACE_DETECTOR_FILENAME,
    ARCFACE_REPO,
    ARCFACE_FILENAME,
)

MODELS_DIR = Path("./.models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_model_from_hf(repo_id: str, filename: str) -> Path:
    """
    Download into a specific folder (MODELS_DIR) instead of the global cache.
    """
    model_path_str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR)
    )
    return Path(model_path_str)

def load_yolo_face_model() -> YOLO:
    """
    Load YOLOv8 face detector weights (PyTorch) using Ultralytics API.
    """
    weights_path = download_model_from_hf(FACE_DETECTOR_REPO, FACE_DETECTOR_FILENAME)
    model = YOLO(str(weights_path))
    return model

def load_arcface_onnx_session() -> ort.InferenceSession:
    """
    Load ArcFace ONNX model into an ONNX Runtime InferenceSession.
    """
    onnx_path = download_model_from_hf(ARCFACE_REPO, ARCFACE_FILENAME)
    # Providers can be tuned; CPUExecutionProvider is safest default.
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    return session

def run_arcface(session: ort.InferenceSession, input_batch: np.ndarray) -> np.ndarray:
    """
    Run ArcFace embedding on a batch of face crops.
    input_batch: (N, H, W, 3) float32, already normalized, NHWC format.
    Returns: (N, D) float32 embeddings.
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_batch})
    return outputs[0]
