# Yolo-Detection: ASL and Sign Digits Detection Project

This project aims to detect American Sign Language (ASL) hand signs and digits using the YOLOv10 model. The process involves working with both **ASL** and **Sign Digits** datasets, performing preprocessing, data augmentation, and model training. The model was initially trained on a combined dataset, and further refinement was made with individual ASL data to improve accuracy.

## Project Overview

### Step-by-Step Process:

1. **Dataset Preparation:**
   - Initially, a combined dataset containing both **ASL** and **Sign Digits** hand sign images was used for training.
   - **Sign Digits images** were originally in grayscale, so we converted them to RGB format for compatibility with the YOLOv10 model.

2. **Data Augmentation and Preprocessing:**
   - The images were resized and augmented to improve the model's ability to generalize.
   - Augmentation techniques such as rotation, flipping, and scaling were applied to the images to simulate real-world variations of hand signs.

3. **Bounding Boxes with MediaPipe:**
   - **MediaPipe** was used to detect and draw bounding boxes around the hand signs in the images.
   - This allowed the model to be trained on more accurately labeled data.

4. **Training the Model on Google Colab:**
   - The initial model was trained on **Google Colab** using the combined dataset.
   - However, the model struggled to differentiate between **ASL** signs and **Digits** signs.

5. **Refining the Model:**
   - To improve performance, we trained **YOLOv10** specifically on the **ASL dataset**.
   - The results were much better, yielding excellent **accuracy** and **precision**.

## Datasets Used

- **Combined ASL and Sign Digits Dataset**: A collection of hand sign images representing both the American Sign Language alphabet and digits (0-9).
- **ASL Dataset**: A dataset containing hand gestures corresponding to letters in the American Sign Language alphabet.
- **Sign Digits Dataset**: A dataset containing hand gestures corresponding to the digits (0-9) in sign language.



