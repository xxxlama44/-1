#3. Feature Extraction

##**3.1. MediaPipe Setup**
"""

# protobuf & tensorflow
!pip install protobuf==4.25.3
!pip install mediapipe==0.10.14
!pip install tensorflow==2.16.1

!pip install --upgrade ml_dtypes

import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5)

"""##**3.2. Helper Functions**"""

#Distance calculation

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

#Angle calculation

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.degrees(np.arccos(cosine_angle))

    return angle

"""##**3.3. Extract Features from Frame**"""

def extract_hand_features(frame):
    # Convert the BGR image to RGB as required by MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the 'hands' object defined in the previous cell
    result = hands.process(rgb)

    # Check if any hands are detected in the frame
    if not result.multi_hand_landmarks:
        return None

    # Retrieve landmarks for the first detected hand (based on max_num_hands=1)
    hand_landmarks = result.multi_hand_landmarks[0]

    # 1. Extract base coordinates (21 landmarks, each with x, y, z)
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    landmarks = hand_landmarks.landmark

    # 2. Calculate additional features (Distances)
    # Get coordinates for Thumb Tip (ID: 4) and Index Tip (ID: 8)
    thumb_tip = [landmarks[4].x, landmarks[4].y]
    index_tip = [landmarks[8].x, landmarks[8].y]

    distance = calculate_distance(thumb_tip, index_tip)

    # 3. Calculate additional features (Angles)
    # Get coordinates for Index Finger joints (IDs: 5, 6, 8) to calculate joint angle
    angle = calculate_angle(
        [landmarks[5].x, landmarks[5].y],
        [landmarks[6].x, landmarks[6].y],
        [landmarks[8].x, landmarks[8].y])

    # Combine all extracted features (Coordinates + Distance + Angle) into a single list
    features = coords + [distance, angle]

    return features

"""##**3.4. Read all videos**"""

BASE_PATH = '/content/drive/MyDrive/مشروع تصميم نظم'

# Define the output directory for extracted features
FEATURES_OUTPUT = os.path.join(BASE_PATH, 'features_csv')
os.makedirs(FEATURES_OUTPUT, exist_ok=True)

# Function to handle complete video processing and feature extraction
def process_video(video_path, label):
    # Initialize video capture from the given path
    cap = cv2.VideoCapture(video_path)
    all_features = []

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If no frame is returned, the video has ended
        if not ret:
            break

        # Extract hand landmarks and calculated features from the current frame
        features = extract_hand_features(frame)

        # If hand features were successfully detected and extracted
        if features is not None:
            # Append the class label (e.g., 'hospital', 'need') to the feature list
            features.append(label)

            # Add the current frame's features to the collection
            all_features.append(features)

    # Release the video capture object to free up system resources
    cap.release()

    return all_features

"""##**3.5. Go through the entire data**"""

# Define the root path for the preprocessed dataset
dataset_path = os.path.join(BASE_PATH, 'processed_videos')

# List to store all extracted features from all videos
all_data = []

# Loop through each dataset split: training, validation, and testing
for split in ['train', 'val', 'test']:

    split_path = os.path.join(dataset_path, split)

    # Check if the split directory exists before proceeding
    if not os.path.exists(split_path):
        print(f"⚠️ Warning: Directory not found: {split_path}")
        continue

    # Iterate through each category folder (e.g., 'bacteria', 'hospital', 'need')
    for label in os.listdir(split_path):

        label_path = os.path.join(split_path, label)

        # Ensure we are processing a directory, not a random file
        if os.path.isdir(label_path):

            # Process each individual video file within the label folder
            for video in os.listdir(label_path):

                # Filter to process only supported video formats (e.g., .mp4)
                if video.lower().endswith(('.mp4', '.mov', '.avi')):
                    video_path = os.path.join(label_path, video)

                    print(f"Processing: {video_path}")

                    # Execute the video processing function for the current video
                    data = process_video(video_path, label)

                    # Add the extracted features to the main data collection
                    all_data.extend(data)

print("✅ Global processing completed. All features are now in 'all_data'.")

dataset_path = os.path.join(BASE_PATH, 'processed_videos')

all_data = []

for split in ['train', 'val', 'test']:

    split_path = os.path.join(dataset_path, split)

    for label in os.listdir(split_path):

        label_path = os.path.join(split_path, label)

        for video in os.listdir(label_path):

            video_path = os.path.join(label_path, video)

            print("Processing:", video_path)

            data = process_video(video_path, label)

            all_data.extend(data)

"""##**3.6. save CSV**"""

import pandas as pd

df = pd.DataFrame(all_data)

csv_path = os.path.join(
    FEATURES_OUTPUT,
    'hand_features.csv')

df.to_csv(csv_path, index=False)

print("Features saved successfully.")
