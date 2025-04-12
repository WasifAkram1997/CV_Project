import os

# ------------------------
# Configuration
# ------------------------
# Replace this with your actual base folder containing subfolders like 'A', 'B', '0', etc.
base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "yolo_images")

# Allowed image extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# ------------------------
# Scan Each Folder
# ------------------------
print(f"\n🔍 Scanning dataset in: {base_folder}\n")

for subfolder in sorted(os.listdir(base_folder)):
    subfolder_path = os.path.join(base_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    total_images = 0
    images_with_labels = 0

    for file_name in os.listdir(subfolder_path):
        if file_name.lower().endswith(image_extensions):
            total_images += 1
            base_name = os.path.splitext(file_name)[0]
            txt_file = os.path.join(subfolder_path, base_name + '.txt')

            if os.path.exists(txt_file):
                images_with_labels += 1

    missing = total_images - images_with_labels
    print(f"📁 {subfolder:<10} | Images: {total_images:<4} | ✅ With .txt: {images_with_labels:<4} | ❌ Missing: {missing}")