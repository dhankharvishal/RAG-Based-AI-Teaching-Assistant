# build_index.py
import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# -------- CONFIG --------
ALL_SEGMENTS = "all_segments.jsonl"   # input (from extract_segments.py)
EMBED_MODEL = "all-MiniLM-L6-v2"      # small, fast, accurate
EMBED_DIM = 384                       # embedding dimension for that model
BATCH_SIZE = 128                      # adjust if memory pressure
INDEX_PATH = "faiss_index.bin"
META_PATH = "index_metadata.pkl"      # stores list of metadata dicts (id->info)
NORMALIZE = True                      # use inner product on normalized vectors (cosine)
# ------------------------

def read_segments(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def build_embeddings(sentences, model):
    # model.encode returns numpy array
    return model.encode(sentences, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)

def main():
    assert Path(ALL_SEGMENTS).exists(), f"{ALL_SEGMENTS} not found."
    print("Loading segments...")
    records = read_segments(ALL_SEGMENTS)
    n = len(records)
    print(f"Loaded {n} segments.")

    print("Loading sentence-transformers model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    # Prepare index
    if NORMALIZE:
        index = faiss.IndexFlatIP(EMBED_DIM)  # inner product on normalized vectors -> cosine
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)  # euclidean

    # Metadata list in same order as vectors in index
    metadata = []

    # Process in batches to avoid memory spikes
    batch_texts = []
    batch_meta = []
    cnt = 0

    print("Encoding & building index (this may take a few minutes)...")
    for rec in tqdm(records, total=n):
        text = rec.get("text", "").strip()
        if not text:
            # skip empty
            metadata.append({"id": rec.get("id"), "video": rec.get("video"), "start": rec.get("start"),
                             "end": rec.get("end"), "text": "", "source_file": rec.get("source_file")})
            # still add a zero vector placeholder? better skip adding empty ones entirely
            continue

        batch_texts.append(text)
        batch_meta.append(rec)

        if len(batch_texts) >= BATCH_SIZE:
            vecs = build_embeddings(batch_texts, model)
            if NORMALIZE:
                faiss.normalize_L2(vecs)
            index.add(vecs)
            metadata.extend(batch_meta)
            cnt += len(batch_texts)
            batch_texts = []
            batch_meta = []

    # last batch
    if batch_texts:
        vecs = build_embeddings(batch_texts, model)
        if NORMALIZE:
            faiss.normalize_L2(vecs)
        index.add(vecs)
        metadata.extend(batch_meta)
        cnt += len(batch_texts)

    print(f"Added {cnt} vectors to the index.")

    # Save index
    print("Saving FAISS index to", INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    print("Saving metadata to", META_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Index build complete. You can now run query_index.py to search.")

if __name__ == "__main__":
    main()
