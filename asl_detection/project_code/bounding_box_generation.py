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
# Create Class Map: A–Z
# ------------------------------------
letter_labels = list(string.ascii_uppercase)
class_names = letter_labels
class_map = {name: idx for idx, name in enumerate(class_names)}

# ------------------------------------
# Initialize MediaPipe Hands
# ------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)  # Improved accuracy with dynamic mode

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

    preview_count = 0  # Counter for the number of previews shown
    for image_file in tqdm(image_files, desc=f"📂 {class_folder}", leave=False):
        if preview_count >= 10:
            break  # Stop after showing 10 previews

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

        # Calculate the min/max coordinates of the bounding box using the hand's outermost points
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # Add padding to the bounding box to ensure the hand is fully captured
        padding = 0.1  # Vertical padding remains the same
        width_padding = 0.2  # Increase padding for width (to make the box wider)

        # Apply padding to the width only
        x_min -= (x_max - x_min) * width_padding
        x_max += (x_max - x_min) * width_padding

        # Apply vertical padding to the height
        y_min -= (y_max - y_min) * padding
        y_max += (y_max - y_min) * padding

        # Apply padding and clamp the coordinates to ensure they stay within the image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # YOLO format (normalized)
        box_w = (x_max - x_min) / w
        box_h = (y_max - y_min) / h
        x_center = (x_min + (x_max - x_min) / 2) / w
        y_center = (y_min + (y_max - y_min) / 2) / h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

        # Drawing bounding box for preview
        image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        image = cv2.putText(image, class_folder, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show image preview
        cv2.imshow(f"Preview: {image_file}", image)
        cv2.waitKey(0)  # Wait for key press to move to the next image
        cv2.destroyAllWindows()

        preview_count += 1  # Increment the preview count

    # After 10 previews, prompt the user to continue or stop
    if preview_count > 0:
        user_input = input(f"Would you like to generate .txt files for {class_folder}? (yes/no): ").strip().lower()
        if user_input == 'yes':
            # Generate the .txt files for the class folder
            for image_file in tqdm(image_files[preview_count:], desc=f"Generating .txt files for {class_folder}"):
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

                # Calculate the min/max coordinates of the bounding box using the hand's outermost points
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # Add padding to the bounding box to ensure the hand is fully captured
                padding = 0.1
                width_padding = 0.2

                x_min -= (x_max - x_min) * width_padding
                x_max += (x_max - x_min) * width_padding
                y_min -= (y_max - y_min) * padding
                y_max += (y_max - y_min) * padding

                # Apply padding and clamp the coordinates to ensure they stay within the image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

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
