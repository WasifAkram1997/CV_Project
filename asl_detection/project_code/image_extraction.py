import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# Set base directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (416, 416)  # Size for YOLOv10 input
output_base = os.path.join(base_dir, "data", "yolo_images")  # New folder for clean, YOLO-ready images

# -----------------------------
# PART 2: Copy ASL Alphabet Dataset
# -----------------------------

asl_source = os.path.join(base_dir, "data", "asl_alphabet")
asl_subfolders = [f for f in os.listdir(asl_source) if os.path.isdir(os.path.join(asl_source, f))]

for folder_name in tqdm(asl_subfolders, desc="Copying ASL alphabet images"):
    src_folder = os.path.join(asl_source, folder_name)
    dst_folder = os.path.join(output_base, folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file_name in image_files:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)

        if not os.path.exists(dst_file):
            try:
                img = Image.open(src_file).convert("RGB")  # Make sure image is RGB
                img = img.resize(IMG_SIZE)  # Resize to YOLO size
                img.save(dst_file)
            except Exception as e:
                print(f"❌ Failed to process image: {src_file}. Error: {e}")

print("✅ All images processed and saved in 'yolo_images' folder, ready for labeling!")