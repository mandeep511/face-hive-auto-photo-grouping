from pathlib import Path
import json

from config import (
    RAW_IMAGES_DIR,
    MAX_IMAGES,
    YOLO_BATCH_SIZE,
    YOLO_BATCH_SIZE_RAM_BUFFER_PERCENT,
    YOLO_BATCH_SIZE_MIN,
    YOLO_BATCH_SIZE_MAX,
)
from models import load_yolo_face_model, load_arcface_onnx_session
from face_detection import list_image_paths, detect_faces_in_images_batch
from batch_size_optimizer import calculate_optimal_batch_size
from embeddings import compute_face_embeddings
from db import (
    get_chroma_collection,
    clear_chroma_collection,
    upsert_face_embeddings,
    get_all_embeddings_and_meta,
)
from clustering import (
    cluster_face_embeddings,
    filter_small_clusters,
    assign_cluster_ids_to_metadatas,
)
from file_ops import create_cluster_folders_and_copy_images


def process_all_images():
    """
    High-level pipeline:
      1) Load models
      2) Scan images
      3) Detect + embed faces, persist to Chroma
      4) Fetch all embeddings
      5) Cluster
      6) Create per-person folders with copies of images
    """
    print(f"[pipeline] RAW_IMAGES_DIR = {RAW_IMAGES_DIR}")
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"RAW_IMAGES_DIR does not exist: {RAW_IMAGES_DIR}")

    # 1) Load models (face detector + ArcFace)
    print("[pipeline] Loading YOLO face detector...")
    yolo_model = load_yolo_face_model()
    print("[pipeline] Loading ArcFace ONNX session...")
    arcface_session = load_arcface_onnx_session()

    # 2) Scan images
    image_paths = list_image_paths(RAW_IMAGES_DIR)
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    print(f"[pipeline] Found {len(image_paths)} images to process")

    # 2.5) Determine optimal batch size
    if YOLO_BATCH_SIZE is None:
        print("[pipeline] Auto-detecting optimal batch size based on available resources...")
        batch_size = calculate_optimal_batch_size(
            ram_buffer_percent=YOLO_BATCH_SIZE_RAM_BUFFER_PERCENT,
            min_batch_size=YOLO_BATCH_SIZE_MIN,
            max_batch_size=YOLO_BATCH_SIZE_MAX,
            prefer_gpu=True,
        )
    else:
        batch_size = YOLO_BATCH_SIZE
        print(f"[pipeline] Using manual batch size: {batch_size}")

    # 3) Detect + embed faces, store in Chroma (batch processing)
    collection = get_chroma_collection()
    # Clear collection at start of each session to reflect current file state
    print("[pipeline] Clearing existing embeddings from Chroma...")
    clear_chroma_collection(collection)
    global_face_counter = 0

    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    print(f"[pipeline] Processing {len(image_paths)} images in {num_batches} batches (batch size: {batch_size})")

    for batch_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        print(f"[pipeline] Batch {batch_num}/{num_batches}: processing {len(batch_paths)} images...")
        
        # Batch face detection
        batch_faces = detect_faces_in_images_batch(yolo_model, batch_paths)
        
        # Process each image's faces
        for img_path, faces in zip(batch_paths, batch_faces):
            if not faces:
                continue
            
            embeddings = compute_face_embeddings(arcface_session, faces)
            if embeddings.shape[0] == 0:
                continue

            ids = []
            metadatas = []
            for local_face_idx, face_dict in enumerate(faces):
                face_id = f"{img_path.stem}_face_{local_face_idx}_{global_face_counter}"
                global_face_counter += 1
                ids.append(face_id)
                metadatas.append(
                    {
                        "image_path": str(img_path),
                        "bbox": json.dumps(face_dict["bbox"]),  # serialize tuple to JSON string
                    }
                )

            upsert_face_embeddings(collection, ids, embeddings, metadatas)

    print(f"[pipeline] Finished upserting faces into Chroma")

    # 4) Fetch all embeddings
    embeddings, metadatas, ids = get_all_embeddings_and_meta(collection)
    print(f"[pipeline] Retrieved {embeddings.shape[0]} face embeddings from Chroma")

    if embeddings.shape[0] == 0:
        print("[pipeline] No embeddings to cluster. Exiting.")
        return

    # 5) Cluster embeddings + filter small clusters
    labels = cluster_face_embeddings(embeddings)
    labels = filter_small_clusters(labels)
    updated_metadatas = assign_cluster_ids_to_metadatas(labels, metadatas)

    # 6) Create organized output folders
    create_cluster_folders_and_copy_images(updated_metadatas)
    print("[pipeline] Done. Check grouped_by_person/ folders inside RAW_IMAGES_DIR.")


def main():
    process_all_images()


if __name__ == "__main__":
    main()
