import os
import cv2
import pickle
import numpy as np
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
detector = MTCNN(keep_all=False, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

DATABASE_PATH = "face_database"
OUTPUT_FILE = "embeddings.pkl"

embeddings = {}

for category in os.listdir(DATABASE_PATH):

    category_path = os.path.join(DATABASE_PATH, category)

    if not os.path.isdir(category_path):
        continue

    for person in os.listdir(category_path):

        person_path = os.path.join(category_path, person)

        if not os.path.isdir(person_path):
            continue

        person_embeddings = []

        for image_name in os.listdir(person_path):

            image_path = os.path.join(person_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face = detector(image)

            if face is None:
                print(f"No face detected in {image_name}")
                continue

            with torch.no_grad():
                embedding = embedder(face.unsqueeze(0).to(device))

            person_embeddings.append(
                embedding.squeeze().cpu().numpy()
            )

        if person_embeddings:
            embeddings[f"{category}/{person}"] = person_embeddings
            print(f"{person}: {len(person_embeddings)} face(s) processed")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"\nEmbeddings saved to {OUTPUT_FILE}")