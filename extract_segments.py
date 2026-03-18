# extract_segments.py
import os
import json
import csv
import time
from pathlib import Path
from typing import List, Dict

# -------- CONFIG --------
TRANSCRIPTS_DIR = "transcripts"        # input folder (contains 120 .json files)
OUT_JSONL = "all_segments.jsonl"       # output JSONL (one JSON per line)
OUT_CSV = "all_segments.csv"           # optional CSV summary
PROGRESS_FILE = "extract_progress.json"  # tracks processed files
SPLIT_LONG_SEGMENTS = True             # True -> split long segments into smaller parts
SPLIT_CHARS = 800                      # max chars per chunk when splitting
OVERLAP_CHARS = 100                    # overlap between chunks
# ------------------------

def list_json_files(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Transcripts folder not found: {folder}")
    return sorted([f for f in p.iterdir() if f.suffix.lower() == ".json"])

def load_progress(path: str) -> Dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": []}

def save_progress(path: str, data: Dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def split_text_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def extract_segments_from_file(path: Path) -> List[Dict]:
    """
    Returns list of records:
    {
      "id": "<video>_<segment_index>[_chunkindex]",
      "video": "<video_basename>",
      "start": <float_seconds or None>,
      "end": <float_seconds or None>,
      "text": "....",
      "source_file": "transcripts/1.mp3.json"
    }
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    base = path.stem  # filename without .json
    records = []

    # Whisper vX usually has "segments" key (list of {start,end,text})
    segments = data.get("segments")
    if segments and isinstance(segments, list):
        for i, seg in enumerate(segments, start=1):
            seg_text = (seg.get("text") or "").strip()
            start = seg.get("start")
            end = seg.get("end")
            if SPLIT_LONG_SEGMENTS and len(seg_text) > SPLIT_CHARS:
                chunks = split_text_chunks(seg_text, SPLIT_CHARS, OVERLAP_CHARS)
                for chunk_i, chunk in enumerate(chunks, start=1):
                    rec = {
                        "id": f"{base}_seg{i}_c{chunk_i}",
                        "video": base,
                        "start": start,
                        "end": end,
                        "text": chunk,
                        "source_file": str(path)
                    }
                    records.append(rec)
            else:
                rec = {
                    "id": f"{base}_seg{i}",
                    "video": base,
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "source_file": str(path)
                }
                records.append(rec)
    else:
        # Fallback: some files may just have "text" without segments
        full_text = (data.get("text") or "").strip()
        duration = data.get("duration")  # may be None
        if full_text:
            if SPLIT_LONG_SEGMENTS:
                chunks = split_text_chunks(full_text, SPLIT_CHARS, OVERLAP_CHARS)
                for j, chunk in enumerate(chunks, start=1):
                    records.append({
                        "id": f"{base}_full_c{j}",
                        "video": base,
                        "start": 0.0,
                        "end": duration,
                        "text": chunk,
                        "source_file": str(path)
                    })
            else:
                records.append({
                    "id": f"{base}_full",
                    "video": base,
                    "start": 0.0,
                    "end": duration,
                    "text": full_text,
                    "source_file": str(path)
                })
    return records

def append_jsonl(path: str, records: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_csv(path: str, records: List[Dict], write_header_if_missing=True):
    need_header = write_header_if_missing and not Path(path).exists()
    with open(path, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["id", "video", "start", "end", "text", "source_file"])
        for r in records:
            writer.writerow([r.get("id"), r.get("video"), r.get("start"), r.get("end"), r.get("text"), r.get("source_file")])

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def main():
    start_all = time.perf_counter()
    files = list_json_files(TRANSCRIPTS_DIR)
    total = len(files)
    print(f"Found {total} transcript files in '{TRANSCRIPTS_DIR}'.")

    progress = load_progress(PROGRESS_FILE)
    processed_files = set(progress.get("processed", []))

    processed_count = 0
    written_segments = 0

    for idx, path in enumerate(files, start=1):
        if str(path.name) in processed_files:
            print(f"[{idx}/{total}] Skipping already processed: {path.name}")
            continue

        t0 = time.perf_counter()
        print(f"[{idx}/{total}] Processing: {path.name} ...", end=" ", flush=True)
        try:
            records = extract_segments_from_file(path)
            if not records:
                print("No segments found (skipping).")
            else:
                append_jsonl(OUT_JSONL, records)
                append_csv(OUT_CSV, records, write_header_if_missing=True)
                written_segments += len(records)
                print(f"wrote {len(records)} segments.")
        except Exception as e:
            print("ERROR:", e)

        processed_files.add(str(path.name))
        progress["processed"] = sorted(list(processed_files))
        save_progress(PROGRESS_FILE, progress)

        t1 = time.perf_counter()
        processed_count += 1
        print(f"Time: {human_time(t1 - t0)}")

    total_time = time.perf_counter() - start_all
    print("="*40)
    print(f"Done. Files processed: {processed_count}/{total}. Segments written: {written_segments}.")
    print(f"Outputs: {OUT_JSONL}  (jsonl),   {OUT_CSV}  (csv)")
    print(f"Progress saved to {PROGRESS_FILE}. Total time: {human_time(total_time)}")

if __name__ == "__main__":
    main()
