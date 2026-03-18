# query_index_minimal.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import textwrap

# ---- CONFIG ----
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_PATH = "faiss_index.bin"
META_PATH = "index_metadata.pkl"
TOP_K = 2               # return exactly two results (best + possible)
NORMALIZE = True
SNIPPET_CHARS = 300     # length of snippet shown for each hit
# -----------------

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    return idx

def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

def embed_query(q, model):
    v = model.encode([q], convert_to_numpy=True)
    if NORMALIZE:
        faiss.normalize_L2(v)
    return v

def format_time(s):
    try:
        if s is None:
            return "00:00"
        total = int(round(s))
        h = total // 3600
        m = (total % 3600) // 60
        sec = total % 60
        if h:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        else:
            return f"{m:02d}:{sec:02d}"
    except Exception:
        return str(s)

def short_snippet(text, max_chars=SNIPPET_CHARS):
    t = text.strip().replace("\n", " ")
    if len(t) <= max_chars:
        return t
    # cut at last full word inside limit
    cut = t[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."

def main():
    print("Loading model and index...")
    model = SentenceTransformer(EMBED_MODEL)
    index = load_index()
    meta = load_meta()
    total = index.ntotal if index is not None else 0
    print(f"Ready. Index vectors: {total}")

    while True:
        q = input("\nQuery (or type 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        t0 = time.perf_counter()
        qvec = embed_query(q, model)
        D, I = index.search(qvec, TOP_K)
        t1 = time.perf_counter()

        # I and D shapes: (1, TOP_K)
        ids = I[0].tolist()
        # minimal output; label first as BEST and second as POSSIBLE (if exists)
        labels = ["BEST MATCH", "POSSIBLE MATCH"]

        print()  # blank line for neatness
        for rank, idx in enumerate(ids):
            if idx < 0:
                continue
            entry = meta[idx]
            label = labels[rank] if rank < len(labels) else f"RESULT {rank+1}"
            vid = entry.get("video")
            start = entry.get("start")
            end = entry.get("end")
            text = entry.get("text", "")
            snippet = short_snippet(text)

            # Minimal, clear output
            print(f"{label}")
            print(f"Video: {vid}   |   {format_time(start)} - {format_time(end)}")
            print(textwrap.fill(snippet, width=100))
            print()  # small spacer

        # if fewer than TOP_K results found, note it (still keep minimal)
        found = sum(1 for idx in ids if idx >= 0)
        if found == 0:
            print("No relevant result found.")
        # tiny timing feedback, optional — kept short
        # print(f"[search time: {int((t1-t0)*1000)} ms]")

if __name__ == "__main__":
    main()
