import os
import requests
import subprocess
import sys
import time

BOT_TOKEN = "8828577506:AAHc9HgklZicytfe_mSqzyU2h4HNYShdmZg"

# Folder where blacklisted faces are stored
BLACKLIST_FOLDER = "face_database/blacklisted"

url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

last_update_id = 0

while True:

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_update_id + 1}"

    updates = requests.get(url).json()["result"]
    

    for update in updates:

        last_update_id = update["update_id"]

        if "message" not in update:
            continue

        message = update["message"]

        if "photo" not in message:
            continue

        if "caption" not in message:
            print("Photo received without caption.")
            continue

        name = message["caption"].strip()

    # Highest quality photo
        file_id = message["photo"][-1]["file_id"]

    # Get file path from Telegram
        file_info = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}"
    ).json()

        file_path = file_info["result"]["file_path"]

    # Download image
        image = requests.get(
        f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    ).content

    # Create folder
        person_folder = os.path.join(BLACKLIST_FOLDER, name)
        os.makedirs(person_folder, exist_ok=True)

        image_name = os.path.join(person_folder, f"{name}.jpg")

        with open(image_name, "wb") as f:
            f.write(image)

        print(f"{name} saved successfully.")
        print("Updating face embeddings...")

        subprocess.run(
    [sys.executable, "generate_embeddings.py"],
    check=True
)

        print("Embeddings updated successfully.")


    # Confirmation message
        chat_id = message["chat"]["id"]

        requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        params={
            "chat_id": chat_id,
            "text": f"✅ {name} added to blacklist successfully."
        }
    )
    time.sleep(2)