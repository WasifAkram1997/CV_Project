import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("/Users/macbookpro/Desktop/CV-Project/CV-Project/project_code/model/best.pt")

# Open MacBook's webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from webcam
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Manually draw bounding boxes and labels
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Get class name
            class_name = model.names[cls] if hasattr(model, "names") else f"Class {cls}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Put label text
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show updated frame
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
