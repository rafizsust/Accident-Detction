import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

# Load YOLO model
model = YOLO('best.pt')

# Function to display mouse cursor position
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse position: ({x}, {y})")

# Create window and set mouse callback
cv2.namedWindow('Mouse Position')
cv2.setMouseCallback('Mouse Position', mouse_event)

# Open video file
cap = cv2.VideoCapture('cr.mp4')

# Load COCO classes
with open("coco_classes.txt", "r") as file:
    class_list = file.read().split("\n")

# Process frames
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Object detection
    results = model.predict(frame)
    boxes = results.xyxy

    # Draw bounding boxes and labels
    for index, row in boxes.iterrows():
        x1, y1, x2, y2, _, d = row
        x1, y1, x2, y2, d = map(int, [x1, y1, x2, y2, d])
        class_name = class_list[d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{class_name}', (x1, y1), 1, 1)

    cv2.imshow("Mouse Position", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

