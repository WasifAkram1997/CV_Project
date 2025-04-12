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
# PART 1: Process Sign Language Digit Dataset
# -----------------------------

X_path = os.path.join(base_dir, "data", "raw_data", "signlanguage_digits_dataset", "Sign-language-digits-dataset", "X.npy")
Y_path = os.path.join(base_dir, "data", "raw_data", "signlanguage_digits_dataset", "Sign-language-digits-dataset", "Y.npy")

X = np.load(X_path, allow_pickle=True)
Y = np.load(Y_path, allow_pickle=True)

# Create folders for digits 0 to 9
for i in range(10):
    os.makedirs(os.path.join(output_base, str(i)), exist_ok=True)

counters = {i: 0 for i in range(10)}

# Save digit images
for idx in tqdm(range(len(X)), desc="Saving digit images"):
    label = int(np.argmax(Y[idx]))  # Find the digit label from the one-hot vector
    folder_path = os.path.join(output_base, str(label))
    counters[label] += 1
    filename = f"{label}{counters[label]}.jpg"

    # Convert grayscale to RGB and resize
    img = Image.fromarray((X[idx] * 255).astype(np.uint8))  # Convert from 0–1 float to 0–255 image
    img = img.convert("RGB")  # Make it 3-channel RGB
    img = img.resize(IMG_SIZE)  # Resize to YOLO-compatible size
    img.save(os.path.join(folder_path, filename))

# -----------------------------
# PART 2: Copy ASL Alphabet Dataset
# -----------------------------

asl_source = os.path.join(base_dir, "data", "raw_data", "asl_alphabet_dataset", "asl_alphabet_train", "asl_alphabet_train")
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