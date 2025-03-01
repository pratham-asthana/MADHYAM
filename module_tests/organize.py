import numpy as np

def create_sequences(video_folder, sequence_length=30, frame_size=(64, 64)):
    frames = sorted(os.listdir(video_folder))
    sequence = []

    for frame_name in frames[:sequence_length]:
        frame_path = os.path.join(video_folder, frame_name)
        frame = cv2.imread(frame_path)
        sequence.append(frame)

    # Pad the sequence if it's shorter than sequence_length
    if len(sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence), *frame_size, 3))
        sequence.extend(padding)

    return np.array(sequence)