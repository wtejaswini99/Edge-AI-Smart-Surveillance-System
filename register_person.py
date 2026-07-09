import cv2
import os
import time

# Create Authorized folder
base_folder = os.path.join("face_database", "authorised")

os.makedirs(base_folder, exist_ok=True)

# Enter person name
name = input("Enter person name: ")

# Folder for this person
person_folder = os.path.join(base_folder, name)

os.makedirs(person_folder, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start camera
cap = cv2.VideoCapture(0)

count = 0
last_capture_time = time.time()

print("\n===== DATA COLLECTION STARTED =====")
print("Capture different variations:")
print("✔ Normal face")
print("✔ Left / Right angle")
print("✔ Up / Down tilt")
print("✔ With scarf")
print("✔ With cap")
print("✔ With glasses")
print("✔ Different expressions")
print("===================================\n")

while count < 80:

    ret, frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve lighting contrast
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        current_time = time.time()

        # Capture every 1 second
        if current_time - last_capture_time >= 1:

            face = gray[y:y+h, x:x+w]

            # Resize for consistency
            face = cv2.resize(face, (200, 200))

            # Blur detection
            blur_value = cv2.Laplacian(face, cv2.CV_64F).var()

            # Save only clear images
            if blur_value > 50:

                count += 1

                image_path = f"{person_folder}/{count}.jpg"

                cv2.imwrite(image_path, face)

                print(f"Captured {count}/80")

                last_capture_time = current_time

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame,
                    f"Images: {count}/80",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Successfully registered {name} with {count} images!")