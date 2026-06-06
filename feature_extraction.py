import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# 1. MediaPipe Setup
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Tracking both hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 2. Helper Functions
# ==========================================
def extract_hand_features(frame):
    """
    Extracts 126 features (63 Left + 63 Right) relative to the wrist landmark.
    Returns a zero array if no hands are detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Initialize empty arrays for 21 landmarks (x, y, z) = 63 features per hand
    lh = np.zeros(63)
    rh = np.zeros(63)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_class in zip(result.multi_hand_landmarks, result.multi_handedness):
            
            # Get coordinates as a 2D array (21 landmarks x 3 axes)
            landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Shift origin to wrist (landmark 0) to make coordinates relative
            wrist = landmarks_3d[0]
            landmarks_relative = landmarks_3d - wrist
            
            # Flatten relative landmarks into a 1D array (63 features)
            landmarks_flattened = landmarks_relative.flatten()

            # Map MediaPipe's Left/Right labels
            if hand_class.classification[0].label == 'Left':
                rh = landmarks_flattened
            else:
                lh = landmarks_flattened

    return np.concatenate([lh, rh]) # Total 126 features

def process_video_to_sequence(video_path):
    """
    Reads a video and samples exactly 30 frames evenly distributed across the entire video duration.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Check if video has enough frames to sample from
    if total_frames >= 30:
        # Pick 30 indices evenly spaced from 0 to total_frames-1
        frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(extract_hand_features(frame))
            else:
                frames.append(np.zeros(126))
    else:
        # Fallback padding for extra short videos
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                frames.append(extract_hand_features(frame))
            else:
                frames.append(np.zeros(126))

    cap.release()
    return np.array(frames)

# ==========================================
# 3. Processing Pipeline
# ==========================================
DATASET_PATH = os.path.join(os.getcwd(), 'processed_videos')

X_data = [] # Stores sequences (30, 126)
y_data = [] # Stores labels

for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(split_path):
        print(f"Directory not found: {split_path}")
        continue

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)

        if os.path.isdir(label_path):
            for video in os.listdir(label_path):
                if video.lower().endswith(('.mp4', '.mov', '.avi')):
                    video_path = os.path.join(label_path, video)

                    print(f"Processing: {split}/{label}/{video}")
                    sequence = process_video_to_sequence(video_path)

                    X_data.append(sequence)
                    y_data.append(label)

# ==========================================
# 4. Save Data Locally
# ==========================================
X_data = np.array(X_data)
y_data = np.array(y_data)

np.save('X_data.npy', X_data)
np.save('y_data.npy', y_data)

print(f"\n Processing complete!")
print(f"Shape of X_data: {X_data.shape}") 
print(f"Shape of y_data: {y_data.shape}")
print("Data saved as X_data.npy and y_data.npy in your root directory.")