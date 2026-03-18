import os

# 👇 Change this path to your actual folder
folder = r"C:\Users\visha\OneDrive\Desktop\final project ds\web_development_videos"

# Get all video files (you can add more extensions if needed)
files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi'))]

# Sort files alphabetically (or by creation time for real order)
files.sort()  # or use: key=lambda f: os.path.getctime(os.path.join(folder, f))

# Loop through and rename with numbering
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    name, ext = os.path.splitext(filename)

    # Create new name with numbering (e.g., 001_Filename.mp4)
    new_name = f"{i:03d}_{name}{ext}"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")

print("✅ All files numbered successfully!")
