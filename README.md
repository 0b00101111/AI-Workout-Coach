# 🏋️ AI Personal Workout Coach

## Overview
The **AI Personal Workout Coach** is a real-time computer vision application designed to assist home workout enthusiasts. Unlike simple rep counters, this system uses a hybrid architecture to analyze **what** you are doing and **how well** you are doing it.

It combines a **Bi-Directional LSTM** (trained on 500+ videos) for action classification with a deterministic **Geometric Spotter** to provide real-time, audio-visual form correction.

## Features
* **Real-Time Classification:** Instantly detects 4 exercises: Squats, Push-ups, Shoulder Presses, and Bicep Curls.
* **Smart Form Correction:**
    * *Squat:* Detects depth (hips below knees) and chest posture.
    * *Press:* Enforces full elbow extension and prevents back arching.
    * *Curl:* Monitors elbow stability (no swinging).
* **Audio Feedback:** Uses Text-to-Speech to give cues like "Go Lower!" or "Extend Arms!" without you needing to look at the screen.
* **Privacy-First:** All processing runs locally on your CPU. No video data is sent to the cloud.

## 📂 Project Structure
```text
AI_Coach_Demo/
├── main.py             # The main application loop (Webcam, UI, Audio)
├── ai_spotter.py       # The geometric logic engine (Form correction rules)
├── best_model.h5       # Pre-trained Bi-LSTM model weights
├── requirements.txt    # Dependency list
└── README.md           # This file
