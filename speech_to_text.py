# transcribe_cpu_json.py
import os
import json
import time
import traceback
import whisper

# CONFIG
AUDIO_DIR = "audio_files"       # Folder with your mp3/mp4/wav files
OUTPUT_DIR = "transcripts"      # Folder where individual JSONs will be saved
PROGRESS_LOG = "progress_log.txt"  # Text log file to track progress
MODEL_NAME = "large-v2"         # Best for Hindi → English
LANG = "hi"                     # 'hi' = Hindi (spoken language)
EXTS = {".mp3", ".wav", ".m4a", ".flac", ".mp4"}  # Allowed audio extensions


def find_audio_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in EXTS:
                files.append(os.path.join(root, fname))
    return files


def human_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def log_progress(message):
    """Write progress updates to progress_log.txt"""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading Whisper model on CPU... please wait, this may take some time.")
    model = whisper.load_model(MODEL_NAME)
    print(f"✅ Model loaded: {MODEL_NAME}")

    files = find_audio_files(AUDIO_DIR)
    if not files:
        print("No audio files found in", AUDIO_DIR)
        return

    total_files = len(files)
    print(f"Found {total_files} audio files.")
    print(f"Transcripts will be saved to: {OUTPUT_DIR}/")

    processed = 0
    total_elapsed = 0.0
    start_run = time.perf_counter()

    for idx, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}.json")

        # If already done, skip
        if os.path.exists(output_path):
            print(f"[{idx}/{total_files}] Skipping (already done): {base}.json")
            continue

        print(f"[{idx}/{total_files}] Transcribing: {base}")
        t0 = time.perf_counter()

        try:
            # Hindi speech → English text
            result = model.transcribe(path, language=LANG, task="translate")

            # Save the result as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            processed += 1
            log_progress(f"✅ Completed {base}.json ({idx}/{total_files})")

        except Exception as e:
            print("❌ Error transcribing", base)
            traceback.print_exc()
            error_data = {
                "error": str(e),
                "model": MODEL_NAME,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            log_progress(f"❌ Error in {base}.json — {e}")

        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_elapsed += elapsed

        avg = total_elapsed / processed if processed else elapsed
        remaining = (total_files - idx) * avg
        print(f"⏱ Time for this file: {human_time(elapsed)} | avg: {human_time(avg)} | est remaining: {human_time(remaining)}")
        print("-" * 40)

    total_time = time.perf_counter() - start_run
    print("✅ All done.")
    print("Total wall-clock time this run:", human_time(total_time))
    print(f"Transcripts saved in: {OUTPUT_DIR}/")

    log_progress(f"✅ Run completed. Total time: {human_time(total_time)}")


if __name__ == "__main__":
    main()
