import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
from collections import deque
import time
import logging
import os

# =========================================================================
# [MEMBER 5: ENVIRONMENT & LIVE MONITORING CONFIGURATION]
# =========================================================================
#os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Setting up automated performance tracing and telemetry pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [AI-Monitor] %(message)s',
    handlers=[
        logging.FileHandler("system_monitor.log"), # Production audit trail
        logging.StreamHandler()                    # Live terminal stdout
    ]
)

# =========================================================================
# 1. Load Trained Model, Scaler, and Configuration
# =========================================================================
logging.info("System Initialization: Loading central model and scaler architectures...")

try:
    # Protected model load using validation custom objects mapping
    model = tf.keras.models.load_model(
        'Emergency_Model_Final.h5',
        custom_objects={
            'LSTM': tf.keras.layers.LSTM,
            'Bidirectional': tf.keras.layers.Bidirectional
        }
    )
    logging.info("Core BiLSTM Model loaded successfully via compatibility layer.")
except Exception as e:
    model = tf.keras.models.load_model('Emergency_Model_Final.h5')
    logging.warning(f"Standard load failed. Switched to direct fallback load. Info: {e}")

scaler = joblib.load('scaler.pkl')

# Configuration matching the training phase
SEQUENCE_LENGTH = 30  # LSTM expects 30 consecutive frames
FEATURE_COUNT = 126   # 63 features for Left Hand + 63 features for Right Hand

# Sliding window mechanism to store the last 30 frames
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

# =========================================================================
# 2. Initialize MediaPipe Hands Tracker
# =========================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Class labels for medical emergency signs (Must match training order)
actions = ['bacteria', 'cancer', 'emergency', 'help', 'hospital', 'need', 'pregnancy', 'swelling', 'virus', 'wound']

# =========================================================================
# [MEMBER 5: STABILITY & SCALABILITY VARIABLES]
# =========================================================================
frame_count = 0
FRAME_SKIP_RATE = 2  # Processes every 2nd frame to cut CPU load by 50%

# =========================================================================
# 3. Real-Time Inference Loop
# =========================================================================
cap = cv2.VideoCapture(0)
logging.info("Hardware Interface Hook: Camera stream opened. System fully operational.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.critical("Hardware Error Hook: Failed to grab frame from camera interface.")
        break

    # 💡 [MEMBER 5 OPTIMIZATION]: Dynamic frame-skipping mechanism to preserve host device RAM/CPU
    frame_count += 1
    if frame_count % FRAME_SKIP_RATE != 0:
        continue

    # Flip frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    text = "Hand not detected"
    hand_detected = False

    # Convert RGB back to BGR for visualization
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize empty templates for both hands (63 zeros each)
    left_hand_features = np.zeros(21 * 3)
    right_hand_features = np.zeros(21 * 3)

    # If hands are detected in the current frame
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_detected = True
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Render landmarks on the screen
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Determine if the detected hand is Left or Right
            handedness = results.multi_handedness[idx].classification[0].label
            
            # Extract relative coordinates matching the feature extraction pipeline
            wrist = hand_landmarks.landmark[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                # Subtract wrist coordinate to maintain translation invariance
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            # Assign features to the corresponding hand template
            if handedness == 'Left':
                left_hand_features = np.array(landmarks)
            elif handedness == 'Right':
                right_hand_features = np.array(landmarks)

    # Combine both hand features into a single frame vector (Shape: 126,)
    current_frame_features = np.concatenate([left_hand_features, right_hand_features])
    
    # Append the current frame vector to our sliding window
    frame_sequence.append(current_frame_features)

    # 💡 Core LSTM Logic: Predict only when we have a complete sequence of 30 frames
    if len(frame_sequence) == SEQUENCE_LENGTH:
        # 1. Convert sequence to numpy array (Shape: 30, 126)
        sequence_array = np.array(frame_sequence)
        
        # 2. Scale features using the saved scaler (Requires 2D input, then reshape back to 3D)
        sequence_scaled = scaler.transform(sequence_array).reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
        
        # 💡 [MEMBER 5 MONITORING HOOK]: Real-Time Latency Calculation
        start_inference_time = time.time()
        
        # 3. Perform optimized feed-forward inference
        predictions = model(sequence_scaled, training=False)[0].numpy()
        
        # Evaluation of runtime execution safety thresholds
        inference_latency = time.time() - start_inference_time
        if inference_latency > 1.0:
            logging.warning(f"Performance Drop Hook! Inference Latency: {inference_latency:.2f}s exceeds 1-second real-time KPI threshold.")
        
        # 4. Extract top prediction details
        predicted_idx = np.argmax(predictions)
        predicted_action = actions[predicted_idx]
        confidence = predictions[predicted_idx]
        print("Prediction:", predicted_action)
        print("Confidence:", confidence)

        # 5. Display the predicted label if confidence passes the 60% threshold
        if confidence > 0.60:
            text = f"SIGN: {predicted_action.upper()} ({confidence*100:.1f}%)"
        else:
            # 💡 [MEMBER 5 MONITORING HOOK]: Low Confidence Anomaly Tracking
            text = "Gesture unclear"
            logging.info(f"Unclear Gesture Tracked: Max prediction confidence was only {confidence*100:.1f}%. Prompting user fallback.")
            
        if not hand_detected:
            # 💡 [MEMBER 5 STABILITY]: Proactive buffer reclamation preventing residual memory leaks
            frame_sequence.clear()
            text = "Hand not detected"
            logging.info("Stability Hook: Hand exited viewport. Frame sequence buffer successfully reclaimed.")
            
        # Draw a professional status bar at the top of the screen
        cv2.rectangle(image, (0, 0), (w, 50), (245, 117, 16), -1)
        cv2.putText(image, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the interactive window
    cv2.imshow('Emergency Sign Language - RealTime Inference', image)

    # Exit smoothly when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("System Termination Request Received. Releasing resources gracefully.")
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
logging.info("System Status: Disconnected. All operational hardware handles freed.")