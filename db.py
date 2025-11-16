# db.py
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import chromadb

from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

def get_chroma_collection() -> "chromadb.Collection":
    """
    Create or load a persistent Chroma collection using the modern PersistentClient API.
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return collection

def clear_chroma_collection(collection: "chromadb.Collection") -> None:
    """
    Clear all embeddings from the Chroma collection.
    This ensures the database reflects the current state of files.
    """
    # Get all IDs and delete them
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

def upsert_face_embeddings(
    collection: "chromadb.Collection",
    ids: List[str],
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    Upsert embeddings + metadata explicitly, without letting Chroma compute embeddings.
    """
    if len(ids) == 0:
        return
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

def get_all_embeddings_and_meta(
    collection: "chromadb.Collection",
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Fetch all embeddings + metadata + ids from the collection for clustering.
    """
    # Chroma get() can be paginated; for simplicity fetch all.
    res = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(res["embeddings"], dtype="float32")
    metadatas = res["metadatas"]
    ids = res["ids"]
    return embeddings, metadatas, ids

# ---- Optional: LangChain wrapper view over the same Chroma DB (not used by pipeline) ----

def create_langchain_chroma_view():
    """
    Optional LangChain integration: create a langchain-chroma VectorStore
    bound to the same persistent directory and collection name.
    This is illustrative; the main pipeline does not depend on it.
    """
    try:
        from langchain_chroma import Chroma as LCChroma  # separate integration package in new LangChain
    except ImportError:
        return None

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    # embedding_function is optional if you don't add/query via LangChain; we only show how to construct.
    vectorstore = LCChroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
    )
    return vectorstore
