# Smart Attendance System (Face Recognition)

Automatic attendance system using face recognition.

## Features
- Known faces encoding (`encode_faces.py`)
- Real-time attendance marking via webcam (`attendance.py`)
- GUI interface with Tkinter (`tk_gui.py`)
- Attendance persisted in `attendance.csv`

## Setup
1. Create virtual env
2. `pip install -r requirements.txt`
3. Put face images into `known_faces/` named `Name.jpg`

## Run
1. `python encode_faces.py`
2. `python attendance.py`
3. `python tk_gui.py` (optional)

## Notes
- `face_recognition` uses dlib; install requirements accordingly.
- For better accuracy, use clean frontal images with decent lighting.

## Bonus ideas
- Emotion detection
- AI resume screening
- Traffic signal control
- Disease prediction from symptoms
- Pose-based AI fitness trainer
