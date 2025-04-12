import os
import shutil
import random
from tqdm import tqdm

# ---------------------
# CONFIG
# ---------------------
source_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "yolo_images")
output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "yolo_dataset")

image_exts = ('.jpg', '.jpeg', '.png')
split_ratio = 0.8  # 80% train, 20% val

# Create destination folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# ---------------------
# Split and Copy
# ---------------------
for class_folder in tqdm(os.listdir(source_dir), desc="📁 Splitting dataset"):
    class_path = os.path.join(source_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Filter image-label pairs
    valid_images = []
    for file_name in os.listdir(class_path):
        if file_name.lower().endswith(image_exts):
            base_name = os.path.splitext(file_name)[0]
            txt_path = os.path.join(class_path, base_name + ".txt")
            if os.path.exists(txt_path):
                valid_images.append(base_name)

    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * split_ratio)
    train_set = valid_images[:split_idx]
    val_set = valid_images[split_idx:]

    for split_name, split_list in [('train', train_set), ('val', val_set)]:
        for base_name in split_list:
            img_src = None
            for ext in image_exts:
                potential = os.path.join(class_path, base_name + ext)
                if os.path.exists(potential):
                    img_src = potential
                    break

            label_src = os.path.join(class_path, base_name + '.txt')

            if img_src and os.path.exists(label_src):
                img_dst = os.path.join(output_dir, 'images', split_name, os.path.basename(img_src))
                label_dst = os.path.join(output_dir, 'labels', split_name, os.path.basename(label_src))

                shutil.copy2(img_src, img_dst)
                shutil.copy2(label_src, label_dst)

print("\n✅ Dataset successfully split into 'train' and 'val' folders.")