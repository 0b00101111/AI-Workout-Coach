# 🏋️ AI Personal Workout Coach

## Overview

The **AI Personal Workout Coach** is a real-time computer vision application designed to assist home workout enthusiasts. Unlike simple rep counters, this system uses a hybrid architecture to analyze **what** you are doing and **how well** you are doing it.

It combines a **Bi-Directional LSTM** (trained on 500+ videos) for action classification with a deterministic **Geometric Spotter** to provide real-time, audio-visual form correction.

## Features

- **Real-Time Classification:** Instantly detects 4 exercises: Squats, Push-ups, Shoulder Presses, and Bicep Curls.
- **Smart Form Correction:**
  - *Squat:* Detects depth (hips below knees) and chest posture.
  - *Press:* Enforces full elbow extension and prevents back arching.
  - *Curl:* Monitors elbow stability (no swinging).
- **Audio Feedback:** Uses Text-to-Speech to give cues like "Go Lower!" or "Extend Arms!" so you don't have to look at the screen.
- **Privacy-First:** All processing runs locally on your CPU. No video data is sent to the cloud.

## 📂 Project Structure

```text
AI_Coach_Demo/
├── main.py             # The main application loop (Webcam, UI, Audio)
├── ai_spotter.py       # The geometric logic engine (Form correction rules)
├── best_model.h5       # Pre-trained Bi-LSTM model weights
├── requirements.txt    # Dependency list
└── README.md           # This file
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 - 3.11
- A working Webcam

### Steps

1. Clone or Unzip this repository.

2. Create a Virtual Environment (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Run

1. Ensure your webcam is connected.

2. Run the application:
   ```bash
   python main.py
   ```

3. **Stand Back:** Make sure your full body (head to feet) is visible in the frame for best accuracy.

4. **Start Working Out:** The AI will automatically detect your exercise and start coaching.
   - *Note:* The system has a "Motion Gate." It will stay idle until you start moving.

5. **Quit:** Press the `Q` key on your keyboard to exit the application.

## ⚠️ Troubleshooting

- **"Camera Error":**
  - *macOS users:* Ensure your Terminal or IDE has permission to access the Camera in `System Settings > Privacy & Security > Camera`.
  - *Windows users:* Check if another app (like Zoom) is using the camera.

- **"Model Not Found":**
  - Ensure `best_model.h5` is in the same folder as `main.py`.

## Credits & Acknowledgments

This project is based on the research and dataset provided by **Riccardo Riccio**.

- **Original Paper:** [Real-Time Fitness Exercise Classification and Counting from Video Frames](https://arxiv.org/html/2411.11548v1)
- **Original Repository:** [Fitness-AI-Trainer](https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting)
- **Dataset:** [Kaggle: Real-time Exercise Recognition](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset)

**Frameworks Used:**

- MediaPipe (Pose Estimation)
- TensorFlow / Keras (LSTM Model)
- OpenCV (Video Processing)
