import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

TEAM = """
### Team Members
- Esraa Al Awam
- Lama Alharbi
- Danah Almaghamsi
- Rawaf Tazi
- Heba Alsulami
"""

model = tf.keras.models.load_model(
    "Emergency_Model_Final.h5",
    custom_objects={
        "LSTM": tf.keras.layers.LSTM,
        "Bidirectional": tf.keras.layers.Bidirectional
    }
)

scaler = joblib.load("scaler.pkl")

classes = [
    "bacteria", "cancer", "emergency", "help", "hospital",
    "need", "pregnancy", "swelling", "virus", "wound"
]

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126

mp_hands = mp.solutions.hands

def extract_features(frame):
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        left_hand_features = np.zeros(63)
        right_hand_features = np.zeros(63)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                wrist = hand_landmarks.landmark[0]

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z
                    ])

                if handedness == "Left":
                    left_hand_features = np.array(landmarks)
                else:
                    right_hand_features = np.array(landmarks)

        return np.concatenate([left_hand_features, right_hand_features])

def predict_video(video_path):
    if video_path is None:
        return "Please record a short webcam video first."
    if isinstance(video_path, dict):
        video_path = video_path["path"]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return "Could not read the video. Please try again."

    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
    frame_sequence = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            features = extract_features(frame)
            frame_sequence.append(features)
        else:
            frame_sequence.append(np.zeros(FEATURE_COUNT))

    cap.release()

    sequence_array = np.array(frame_sequence)
    sequence_scaled = scaler.transform(sequence_array).reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)

    predictions = model(sequence_scaled, training=False)[0].numpy()
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    if confidence < 0.60:
        return f"Gesture unclear. Confidence: {confidence * 100:.1f}%"

    return f"Predicted Gesture: {classes[predicted_index].upper()} ({confidence * 100:.1f}%)"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink")) as demo:
    gr.Markdown("""
    # 🚨 Emergency Arabic Sign Language Support System
    AI-powered emergency gesture recognition using MediaPipe + BiLSTM

    Record a short webcam video, then click *Start Detection*.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## 🧠 System Pipeline
            Webcam → MediaPipe → 21 Landmarks → 126 Features → 30 Frames → BiLSTM → Prediction
            """)

            video_input = gr.Video(
                sources=["webcam"],
                label="Record Emergency Gesture"
            )

            start_btn = gr.Button("▶️ Start Detection", variant="primary")
            output = gr.Textbox(label="Prediction Result")

        with gr.Column():
            gr.Markdown("""
            ## ✅ Supported Emergency Gestures
            bacteria • cancer • emergency • help • hospital  
            need • pregnancy • swelling • virus • wound
            """)

            gr.Markdown(TEAM)

    start_btn.click(fn=predict_video, inputs=video_input, outputs=output)

demo.launch(share=True)