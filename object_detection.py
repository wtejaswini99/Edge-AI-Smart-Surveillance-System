import cv2
from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolov8n.pt")

def detect_objects(frame):

    results = model(frame)

    annotated_frame = results[0].plot()

    person_count = 0

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        label = model.names[class_id]

        if label == "cell phone":
            cv2.putText(
        annotated_frame,
        "ALERT: Mobile Phone Detected!",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

        if label == "person":
            person_count += 1

        if person_count > 1:
            cv2.putText(
        annotated_frame,
        "ALERT: Multiple People Detected!",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (600, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = f"alerts/multiple_people_{timestamp}.jpg"

    cv2.imwrite(image_path, annotated_frame)

    return annotated_frame