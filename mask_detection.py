import cv2
from ultralytics import YOLO

model = YOLO("ppe_detector_best.pt")

def detect_mask(frame):

    results = model(frame)

    annotated_frame = results[0].plot()

    mask_detected = False

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        label = model.names[class_id]
        confidence = float(box.conf[0])

        print("Label:", label, "Confidence:", confidence)
        print("detected:", label)

        if label == "Mask":

            mask_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.putText(
                annotated_frame,
                "Checking Face Mask...",
                (x2 - 220, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

    print("mask_detected =", mask_detected)

    return annotated_frame, mask_detected