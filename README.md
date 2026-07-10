# Edge AI Smart Surveillance System

## Overview

Edge AI Smart Surveillance System is an intelligent real-time surveillance application developed using Python and computer vision techniques. The system captures live video through a webcam and performs face recognition, person detection, and mask detection on edge devices without relying on cloud processing. It also records surveillance events in an SQLite database, displays them on a Flask web dashboard, and sends instant Telegram alerts whenever an unknown person is detected.

---

## Features

- Real-time face recognition using FaceNet and MTCNN
- Unknown person detection
- Person detection using YOLOv8
- Face mask detection
- Automatic Telegram alert notifications with captured images
- SQLite database for storing surveillance records
- Flask-based web dashboard for viewing detection history
- Face registration and embedding generation for new users
- Edge AI processing without cloud dependency

---

## Technologies Used

- Python
- OpenCV
- FaceNet
- MTCNN
- YOLOv8
- Flask
- SQLite
- NumPy
- PyTorch
- Requests

---

## Project Structure

```
Edge-AI-Smart-Surveillance-System/
│
├── final_facenet_surveillance.py
├── register_person.py
├── generate_embeddings.py
├── object_detection.py
├── mask_detection.py
├── database.py
├── server.py
├── telegram_receiver.py
├── templates/
│   └── dashboard.html
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/wtejaswini99/Edge-AI-Smart-Surveillance-System.git
```

2. Move into the project directory

```bash
cd Edge-AI-Smart-Surveillance-System
```

3. Install the required libraries

```bash
pip install -r requirements.txt
```

4. Create a `config.py` file

```python
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

5. Download the required model files and place them in the project folder.

6. Run the main surveillance system

```bash
python final_facenet_surveillance.py
```

---

## Future Improvements

- Blacklisted person management
- Multi-camera surveillance
- Weapon detection
- Cloud database integration
- Mobile application support
- Email notifications
- Face recognition accuracy improvements

---

## Author

**W. Tejaswini**

Master of Computer Applications (MCA)

Project: Edge AI Smart Surveillance System




