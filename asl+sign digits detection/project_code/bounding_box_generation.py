import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import string

# ------------------------------------
# Configuration
# ------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "yolo_images")

# ------------------------------------
# Create Class Map: 0–9, A–Z, del, nothing, space
# ------------------------------------
digit_labels = [str(i) for i in range(10)]
letter_labels = list(string.ascii_uppercase)
extra_labels = ['del', 'nothing', 'space']
class_names = digit_labels + letter_labels + extra_labels
class_map = {name: idx for idx, name in enumerate(class_names)}

# ------------------------------------
# Initialize MediaPipe Hands
# ------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ------------------------------------
# Loop through class folders
# ------------------------------------
for class_folder in tqdm(os.listdir(IMAGE_DIR), desc="🔍 Processing class folders"):
    class_path = os.path.join(IMAGE_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    if class_folder not in class_map:
        print(f"⚠️ Skipping unknown class folder: {class_folder}")
        continue

    class_id = class_map[class_folder]
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in tqdm(image_files, desc=f"📂 {class_folder}", leave=False):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            continue  # No hand detected

        # Get coordinates of all landmarks
        x_coords = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
        y_coords = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]

        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))

        # YOLO format (normalized)
        box_w = (x_max - x_min) / w
        box_h = (y_max - y_min) / h
        x_center = (x_min + (x_max - x_min) / 2) / w
        y_center = (y_min + (y_max - y_min) / 2) / h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

        # Save annotation
        txt_filename = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(class_path, txt_filename)
        with open(txt_path, "w") as f:
            f.write(yolo_line)

print("✅ YOLO annotations created and saved using MediaPipe hand detection.")