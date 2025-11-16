from pathlib import Path

RAW_IMAGES_DIR = Path("./my-photos")  # input
OUTPUT_BASE_DIR = RAW_IMAGES_DIR / "grouped_by_person"  # output root
CHROMA_DB_DIR = Path("./.chroma_faces")

CHROMA_COLLECTION_NAME = "faces"

# Hugging Face repositories / filenames
# You must check the actual filenames in the repos and adjust as needed.
FACE_DETECTOR_REPO = "arnabdhar/YOLOv8-Face-Detection"  # YOLOv8 face model repo
FACE_DETECTOR_FILENAME = "model.pt"              # example filename; verify on HF

ARCFACE_REPO = "garavv/arcface-onnx"                   # ArcFace ONNX repo
ARCFACE_FILENAME = "arc.onnx"                        # example filename; verify on HF

# Detection / embedding parameters
YOLO_CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES = None           # set to an int for quick testing
YOLO_BATCH_SIZE = None      # None = auto-detect based on available RAM/GPU, or set to int for manual override
YOLO_BATCH_SIZE_RAM_BUFFER_PERCENT = 10.0  # Percentage of RAM to keep free (default 10%)
YOLO_BATCH_SIZE_MIN = 1     # Minimum batch size
YOLO_BATCH_SIZE_MAX = 64    # Maximum batch size
CLUSTER_DISTANCE_THRESHOLD = 0.65  # cosine distance for clustering
MIN_FACES_PER_CLUSTER = 3
