import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

# Function to copy images from ASL Alphabet Dataset to combined data
def combine_asl_alphabet(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for foldername in tqdm(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, foldername)
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_dir, foldername)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Copy images from the ASL Alphabet dataset
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(folder_path, filename)
                    shutil.copy(img_path, output_folder)

# Function to combine the Sign Language Digits Dataset (npy format)
def combine_digits(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_dir, filename)
            data = np.load(file_path)

            # Convert images to the required format
            for i, img in enumerate(data):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Ensure it's RGB
                output_path = os.path.join(output_dir, f"{filename}_{i}.jpg")
                cv2.imwrite(output_path, img_rgb)

# Main function to combine both datasets
def combine_datasets():
    # Updated paths based on your directory structure
    asl_alphabet_dir = 'data/raw_data/asl_alphabet_dataset'  # Correct path for ASL alphabet dataset
    signlanguage_digits_dir = 'data/raw_data/signlanguage_digits_dataset'  # Correct path for Sign Language Digits dataset
    combined_data_dir = 'data/combined_data/images'  # Combined data directory

    # Combine both datasets into the unified folder
    combine_asl_alphabet(asl_alphabet_dir, combined_data_dir)
    combine_digits(signlanguage_digits_dir, combined_data_dir)

    print("✅ Datasets have been combined successfully!")

if __name__ == "__main__":
    combine_datasets()