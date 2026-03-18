import os
import subprocess

# ---------- CONFIG ----------
source_folder = r"C:\Users\visha\OneDrive\Desktop\final project ds\web_development_videos"
output_folder = os.path.join(source_folder, "audio_files")
# ----------------------------

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all .mp4 files in source folder
files = [f for f in os.listdir(source_folder) if f.lower().endswith(".mp4")]

# Loop through and convert each file
for filename in files:
    video_path = os.path.join(source_folder, filename)
    audio_filename = os.path.splitext(filename)[0] + ".mp3"
    audio_path = os.path.join(output_folder, audio_filename)

    # FFmpeg command (high quality, silent output)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # no video
        "-acodec", "libmp3lame", # use MP3 codec
        "-q:a", "2",             # audio quality (0=best, 9=worst)
        audio_path,
        "-y"                     # overwrite if exists
    ]

    print(f"🎵 Converting: {filename} → {audio_filename}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\n✅ All videos converted to audio successfully!")
print(f"Audio files saved in: {output_folder}")
