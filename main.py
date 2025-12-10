# ==========================================
# FINAL POLISHED VERSION: main.py
# ==========================================
print("STEP 1: Importing libraries...")
import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import subprocess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization

# --- AUDIO ENGINE ---
def speak_worker(text):
    subprocess.run(["say", text])

def speak(text):
    threading.Thread(target=speak_worker, args=(text,), daemon=True).start()

# --- MODEL BUILDER ---
def build_local_model():
    input_shape = (30, 99)
    num_classes = 4
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- MOTION GATE ---
def calculate_motion(buffer):
    if len(buffer) < 5: return 0.0
    start_pose = np.mean(buffer[:5], axis=0)
    end_pose = np.mean(buffer[-5:], axis=0)
    return np.mean(np.abs(end_pose - start_pose))

# --- MAIN APP ---
def main():
    print("\nSTEP 2: Building Model...")
    try:
        from ai_spotter import AISpotter
        model = build_local_model()
        model.load_weights('best_model.h5')
        print("✅ Model Weights Loaded!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    spotter = AISpotter()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    sequence_buffer = []
    CLASSES = ['barbell biceps curl', 'push-up', 'shoulder press', 'squat']

    # --- SETTINGS ---
    last_speech_time = 0
    COOLDOWN_SECONDS = 4.0
    MOTION_THRESHOLD = 0.035

    print("\nSTEP 3: Camera Starting... Press 'Q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera Error.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        label = "Idle (Stand Still)"
        conf_text = ""
        feedback = ""
        box_color = (50, 50, 50)
        is_moving = False
        motion_score = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cx = (landmarks[23].x + landmarks[24].x) / 2
            cy = (landmarks[23].y + landmarks[24].y) / 2
            cz = (landmarks[23].z + landmarks[24].z) / 2

            row = []
            for lm in landmarks:
                row.extend([lm.x - cx, lm.y - cy, lm.z - cz])

            sequence_buffer.append(row)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)

            # --- MOTION CHECK ---
            motion_score = calculate_motion(sequence_buffer)
            is_moving = motion_score > MOTION_THRESHOLD

            if len(sequence_buffer) == 30:
                if is_moving:
                    input_data = np.array([sequence_buffer])
                    prediction = model.predict(input_data, verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > 0.7:
                        label = CLASSES[class_id]
                        conf_text = f"{int(confidence*100)}%"

                        # Get 3-State Feedback
                        feedback, box_color, is_perfect = spotter.analyze_frame(label, landmarks)

                        # --- VOICE LOGIC (UPDATED) ---
                        current_time = time.time()
                        if current_time - last_speech_time > COOLDOWN_SECONDS:
                            if feedback:
                                # Case 1: Error detected -> "Shoulder Press, Extend Arms!"
                                speak(f"{label}, {feedback}")
                                last_speech_time = current_time

                            elif is_perfect:
                                # Case 2: Perfect Form -> "Good Shoulder Press!"
                                speak(f"Good {label}")
                                last_speech_time = current_time

                            else:
                                # Case 3: Neutral (Mid-rep) -> SILENCE
                                pass
                    else:
                        label = "Uncertain"

        # --- UI OVERLAY ---
        h, w, _ = frame.shape

        # 1. Top Bar
        status_color = (0, 200, 0) if is_moving else (100, 100, 100)
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"{label} {conf_text}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Motion Meter
        bar_width = int(min(motion_score / (MOTION_THRESHOLD * 2), 1.0) * 200)
        cv2.rectangle(frame, (w-220, 10), (w-20, 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (w-220, 10), (w-220+bar_width, 30), status_color, -1)
        cv2.line(frame, (w-120, 5), (w-120, 35), (0, 0, 255), 2) # Threshold
        cv2.putText(frame, "Motion", (w-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 3. Feedback Banner
        if feedback and is_moving:
            cv2.rectangle(frame, (0, h-70), (w, h), box_color, -1)
            cv2.putText(frame, feedback, (50, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('AI Personal Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()