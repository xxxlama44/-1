INPUT_PATH = os.path.join(BASE_PATH, 'trimmed_videos')
OUTPUT_PATH = os.path.join(BASE_PATH, 'processed_videos')

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FPS = 30

os.makedirs(OUTPUT_PATH, exist_ok=True)

def preprocess_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open:", input_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        TARGET_FPS,
        (TARGET_WIDTH, TARGET_HEIGHT)
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # Save standardized frame
        out.write(frame)

    cap.release()
    out.release()


for split in ['train', 'val', 'test']:
    split_input_path = os.path.join(INPUT_PATH, split)
    split_output_path = os.path.join(OUTPUT_PATH, split)

    if not os.path.exists(split_input_path):
        print("Missing split:", split_input_path)
        continue

    os.makedirs(split_output_path, exist_ok=True)

    for label in os.listdir(split_input_path):
        label_input_path = os.path.join(split_input_path, label)
        label_output_path = os.path.join(split_output_path, label)

        if not os.path.isdir(label_input_path):
            continue

        os.makedirs(label_output_path, exist_ok=True)

        for video_file in os.listdir(label_input_path):

            if not video_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                continue

            input_video_path = os.path.join(label_input_path, video_file)

            output_video_name = os.path.splitext(video_file)[0] + ".mp4"
            output_video_path = os.path.join(label_output_path, output_video_name)

            preprocess_video(input_video_path, output_video_path)

            print(f"Processed: {split}/{label}/{output_video_name}")

print("All trimmed videos have been preprocessed successfully.")
print(f"{'Folder (Label)':<20} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
print("-" * 50)

all_labels = set()

for split in ['train', 'val', 'test']:
    split_path = os.path.join(OUTPUT_PATH, split)
    if os.path.exists(split_path):
        all_labels.update(os.listdir(split_path))

for label in sorted(all_labels):
    counts = []

    for split in ['train', 'val', 'test']:
        label_path = os.path.join(OUTPUT_PATH, split, label)

        if os.path.exists(label_path):
            videos = [
                f for f in os.listdir(label_path)
                if f.lower().endswith('.mp4')
            ]
            counts.append(len(videos))
        else:
            counts.append(0)

    print(f"{label:<20} | {counts[0]:<7} | {counts[1]:<7} | {counts[2]:<7}")

print("-" * 50)
print("Processed dataset verification completed.")
