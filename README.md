# FaceHive - Automatic Photo Grouping 📸

Automatically group photos by detecting and clustering faces. No manual tagging, no metadata dependencies—just raw computer vision and some math.

## What It Does

Scans your photo directory, detects faces with YOLOv8, generates ArcFace embeddings, clusters them using agglomerative clustering, and organizes photos into person-specific folders. The entire pipeline runs in a single pass with automatic batch size optimization.

## Architecture

- **Face Detection**: [YOLOv8 Face Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection) with configurable confidence threshold
- **Embeddings**: [ArcFace ONNX](https://huggingface.co/garavv/arcface-onnx) model (512D normalized vectors)
- **Clustering**: Agglomerative clustering with cosine distance threshold
- **Storage**: ChromaDB for vector storage and retrieval
- **Batch Processing**: Auto-optimized batch sizes based on available RAM/GPU

The pipeline is stateless—each run clears the database and rebuilds from scratch, so deleted or moved files don't leave stale embeddings.

## Quick Start

```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Point config.py to your photo directory, then:
uv run python main.py
```

Results appear in `grouped_by_person/` with folders named `person_000`, `person_001`, etc. ✨

## Configuration

Edit `config.py`:

- `RAW_IMAGES_DIR`: Source photo directory
- `YOLO_CONFIDENCE_THRESHOLD`: Face detection sensitivity (default: 0.3)
- `CLUSTER_DISTANCE_THRESHOLD`: Cosine distance for grouping (default: 0.65)
- `MIN_FACES_PER_CLUSTER`: Minimum faces per person (default: 3)
- `YOLO_BATCH_SIZE`: Manual override or `None` for auto-detection

## Technical Details

- **Embedding Normalization**: L2-normalized 512D vectors
- **Clustering Metric**: Cosine distance with average linkage
- **File Handling**: Preserves directory structure in output filenames to prevent overwrites

Models are downloaded from HuggingFace on first run and cached in `.models/`. ChromaDB data lives in `.chroma_faces/`.

## Requirements

- Python 3.13+ (probably works with 3.10+)
- ~2GB RAM minimum (auto-scales batch size)
- Models download automatically (~500MB total)

## License

MIT
