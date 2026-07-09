# ==========================================================
# EDGE AI SMART SURVEILLANCE SYSTEM
#
# Features:
# 1. Face Recognition
# 2. Unknown Person Detection
# 3. Mask Detection
# 4. Weapon Detection
# 5. Object Detection
# 6. Telegram Alerts
# 7. Alarm System
# 8. Alert Logging
# ==========================================================
import cv2
import os
import time
import pickle
import requests
import sqlite3
import winsound
import numpy as np
import torch


from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1

from mask_detection import detect_mask
from object_detection import detect_objects
from database import save_detection
from config import BOT_TOKEN,CHAT_ID

# Telegram configurationexit
BOT_TOKEN = "BOT_TOKEN"
CHAT_ID = "CHAT_ID"


# Send Telegram text message
def send_telegram_alert(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:

        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=5
        )

    except Exception as e:

        print("Telegram Message Error:", e)


# Send Telegram photo
def send_telegram_photo(photo_path, caption):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:

        with open(photo_path, "rb") as photo:

            requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption
                },
                files={
                    "photo": photo
                },
                timeout=5
            )

    except Exception as e:

        print("Telegram Photo Error:", e)


# Load FaceNet embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)


# Load FaceNet embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
print("People:", embeddings.keys())
print("Number of people:", len(embeddings))


device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

embedder = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


# Load weapon detection model
weapon_model = YOLO("weapon_detection.pt")


# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# SQLite Database
conn = sqlite3.connect("surveillance.db")
cursor = conn.cursor()

# Create alerts folder
if not os.path.exists("alerts"):
    os.makedirs("alerts")


# Variables
unknown_counter = 0
mask_alert_sent = False
last_db_save_time = 0
last_weapon_alert_time = 0
last_intruder_alert_time = 0
frame_count = 0
mask_detected = False
last_saved_person = ""
last_saved_time = 0
image_path = ""
last_blacklist_alert = 0

# ===========================
# Main Camera Loop
# ===========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera Error!")
        break

    frame_count += 1

    # ===========================
    # Face Recognition (FaceNet)
    # ===========================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))

            face = face.astype(np.float32)

            face_tensor = torch.from_numpy(face).permute(2, 0, 1)
            face_tensor = (face_tensor / 255.0)
            face_tensor = face_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = embedder(face_tensor).cpu().numpy()[0]

            best_name = "Unknown"
            best_distance = 999

            for person, vectors in embeddings.items():

                for saved_vector in vectors:

                    distance = np.linalg.norm(embedding - saved_vector)

                    if distance < best_distance:
                        best_distance = distance
                        best_name = person

            if best_distance < 0.85:

                if "/" in best_name:

                    category, person_name = best_name.split("/", 1)

                    if category.lower() == "authorized":
                        name = f"Authorized : {person_name}"
                        color = (0, 255, 0)

                    elif category.lower() == "blacklisted":
                        name = f"Blacklisted : {person_name}"
                        color = (0, 0, 255)

                        current_time = time.time()

                        if current_time - last_blacklist_alert > 30:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                image_path = f"alerts/blacklisted_{timestamp}.jpg"
                                cv2.imwrite(image_path, frame)

                                send_telegram_photo(
                            image_path,
                            f"🚨 ALERT!\nBlacklisted person detected!\nName: {person_name}"
    )
                                save_detection(
    f"Blacklisted : {person_name}",
    "Yes" if mask_detected else "No",
    best_distance,
    image_path
)
                                last_blacklist_alert = current_time

                    else:
                        name = person_name
                        color = (255, 255, 255)

                else:
                    name = best_name
                    color = (0, 255, 0)

                unknown_counter = 0

            else:
                name = "Unknown"
                color = (0, 255, 255)
                unknown_counter += 1

                if unknown_counter >= 10:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"alerts/intruder_{timestamp}.jpg"

                    cv2.imwrite(image_path, frame)

                    send_telegram_photo(
                    image_path,
            "🚨     Unknown person detected!"
            )

                    unknown_counter = 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        print(f"{name} -> Distance: {best_distance:.3f}")
        
        current_time = time.time()
        if (name != last_saved_person) or (current_time - last_saved_time > 5):
                image_file = ""
                if name == "Unknown":
                    image_file = image_path

                save_detection(
                name,
                "Yes" if mask_detected else "No",
                best_distance,
        image_file
    )

                last_saved_person = name
                last_saved_time = current_time
                

        cv2.putText(
                frame,
                f"{name} ({best_distance:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
   
    # Object Detection
    if frame_count % 3 == 0 :
        frame = detect_objects(frame)

    # Mask Detection
    if frame_count % 3 == 0 :
        frame, mask_detected = detect_mask(frame)

    if not mask_detected:
        mask_alert_sent = False

    if mask_detected and not mask_alert_sent:

        timestamp = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
        image_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = f"alerts/mask_{image_name}.jpg"

        cv2.imwrite(image_path, frame)

        send_telegram_alert(
                f"😷 MASK ALERT\n\n"
                f"Masked person detected.\n"
                f"Time: {timestamp}"
            )

        send_telegram_photo(
                image_path,
                f"😷 MASK DETECTED\n\nTime: {timestamp}"
            )

        mask_alert_sent = True
    # Weapon Detection  
    if frame_count % 3 == 0 :
        weapon_results = weapon_model(frame, conf=0.80)

        for result in weapon_results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                weapon_name = weapon_model.names[class_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.putText(
                    frame,
                    f"{weapon_name} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                current_time = time.time()

                if current_time - last_weapon_alert_time > 30:

                    timestamp = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

                    send_telegram_alert(
                        f"🚨 WEAPON DETECTED\n\n"
                        f"Weapon : {weapon_name}\n"
                        f"Confidence : {confidence:.2f}\n"
                        f"Time : {timestamp}"
                    )

                    last_weapon_alert_time = current_time
    # Intruder Detection
    if unknown_counter >= 3:
        current_time = time.time()

        if current_time - last_intruder_alert_time > 30:

            timestamp = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
            image_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            image_path = f"alerts/intruder_{image_name}.jpg"

            cv2.imwrite(image_path, frame)

            send_telegram_alert(
                f"🚨 INTRUDER ALERT\n\n"
                f"Unknown person detected.\n"
                f"Time : {timestamp}"
            )

            send_telegram_photo(
                image_path,
                f"🚨 INTRUDER PHOTO\n\nTime : {timestamp}"
            )

            winsound.Beep(1000,300)
            winsound.Beep(1500,300)

            with open("alerts/log.txt","a") as log:
                log.write(f"Unknown detected at {timestamp}\n")

            last_intruder_alert_time = current_time
            unknown_counter = 0
    # Display Video
    cv2.imshow("Edge AI Smart Surveillance", frame)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()