
# Arabic Sign Language for Emergency (ASL-Emergency)

A computer vision-based system designed to recognize 10 essential Arabic sign language gestures for medical emergencies. The system translates these gestures into actionable commands in real-time.

## Features
- **Real-time Recognition:** Captures and processes hand gestures instantly.
- **Emergency Focused:** Dedicated to 10 critical medical signs.
- **High Precision:** Utilizes advanced skeletal tracking for accuracy.

## Tech Stack
- **Language:** Python
- **Vision:** MediaPipe (for hand landmark detection)
- **AI/ML:** BiLSTM (Bidirectional Long Short-Term Memory) for temporal sequence analysis.

## How it works
1. Captures live video input.
2. Extracts hand landmarks using MediaPipe.
3. Passes the sequence to the BiLSTM model to classify the emergency sign.

## Project Status
Completed as part of a university project focusing on AI for social good.
