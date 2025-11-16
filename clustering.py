from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import CLUSTER_DISTANCE_THRESHOLD, MIN_FACES_PER_CLUSTER

def cluster_face_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Agglomerative clustering with a distance threshold, letting sklearn
    determine number of clusters automatically.
    Returns: labels (N,) with -1 for noise/singletons.
    """
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
    )
    labels = clustering.fit_predict(embeddings)
    return labels

def filter_small_clusters(labels: np.ndarray) -> np.ndarray:
    """
    Set labels for clusters smaller than MIN_FACES_PER_CLUSTER to -1 (ignored).
    """
    if labels.size == 0:
        return labels
    filtered = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    for lab, count in zip(unique, counts):
        if count < MIN_FACES_PER_CLUSTER:
            filtered[labels == lab] = -1
    return filtered

def assign_cluster_ids_to_metadatas(
    labels: np.ndarray,
    metadatas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add 'cluster_id' to each metadata dict.
    """
    updated = []
    for label, meta in zip(labels, metadatas):
        m = dict(meta)
        m["cluster_id"] = int(label)
        updated.append(m)
    return updated
