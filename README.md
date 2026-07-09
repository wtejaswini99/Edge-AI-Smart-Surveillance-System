# Edge AI Smart Surveillance System

## Overview

The Edge AI Smart Surveillance System is an intelligent real-time surveillance application developed using Python, Computer Vision, and Deep Learning. The system performs face recognition, object detection, mask detection, and generates instant Telegram alerts for suspicious activities. It stores detection records in an SQLite database and provides a Flask-based dashboard for monitoring surveillance events.

## Features

- Real-time face recognition using FaceNet and MTCNN
- Authorized and blacklisted person identification
- Unknown person detection
- Object detection using YOLOv8
- Face mask detection
- Telegram alert notifications with captured images
- SQLite database for detection records
- Flask web dashboard for surveillance monitoring

## Technologies Used

- Python
- OpenCV
- FaceNet
- MTCNN
- YOLOv8
- PyTorch
- Flask
- SQLite
- Telegram Bot API

## Project Structure

```
Edge-AI-Smart-Surveillance/
├── final_facenet_surveillance.py
├── register_person.py
├── generate_embeddings.py
├── mask_detection.py
├── object_detection.py
├── database.py
├── server.py
├── telegram_receiver.py
├── templates/
└── README.md
```

## Installation

1. Clone the repository.
2. Install the required Python libraries.
3. Create a `config.py` file containing:

```python
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

4. Generate face embeddings.
5. Run the surveillance system.

## Author

**Wupadrastha Tejaswini**

Master of Computer Applications (MCA)

Project: **Edge AI Smart Surveillance System**