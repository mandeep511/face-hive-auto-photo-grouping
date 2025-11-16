from pathlib import Path
from typing import List, Dict, Any, Set
import shutil

from config import OUTPUT_BASE_DIR, RAW_IMAGES_DIR

def ensure_output_root() -> Path:
    """
    Ensure the base output directory exists and return it.
    """
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_BASE_DIR

def create_cluster_folders_and_copy_images(
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    For each cluster_id >= 0, create a folder 'person_{cluster_id}' and copy
    the corresponding image files into it (no dedup across clusters).
    Expect metadatas to contain: 'image_path' (str) and 'cluster_id' (int).
    """
    output_root = ensure_output_root()
    # Map cluster_id -> set of image paths so each image is copied once per cluster.
    clusters: Dict[int, Set[str]] = {}
    for meta in metadatas:
        cid = meta.get("cluster_id", -1)
        if cid < 0:
            continue
        img_path = meta["image_path"]
        clusters.setdefault(cid, set()).add(img_path)

    for cid, paths in clusters.items():
        cluster_dir = output_root / f"person_{cid:03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for src_str in paths:
            src = Path(src_str)
            if not src.exists():
                continue
            # Create unique filename that preserves directory structure to avoid overwrites
            try:
                # Get relative path from RAW_IMAGES_DIR
                rel_path = src.relative_to(RAW_IMAGES_DIR)
                # Replace path separators with underscores to create unique filename
                unique_name = str(rel_path).replace("/", "_").replace("\\", "_")
            except ValueError:
                # If path is not relative to RAW_IMAGES_DIR, fall back to original name
                # but add a hash of the full path to make it unique
                unique_name = f"{src.stem}_{hash(src_str) % 100000:05d}{src.suffix}"
            dst = cluster_dir / unique_name
            # Copy instead of move so original library stays unchanged.
            shutil.copy2(src, dst)
