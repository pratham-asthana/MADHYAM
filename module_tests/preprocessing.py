import cv2
import os

def extract_frames(video_path, output_folder, frame_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = frame / 255.0  # Normalize
        save_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame * 255)  # Save as image
        frame_count += 1

    cap.release()