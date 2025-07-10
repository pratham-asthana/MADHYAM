import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

FRAME_SIZE = (64, 64)  
SEQUENCE_LENGTH = 30  
NUM_CLASSES = 47 
BATCH_SIZE = 32
EPOCHS = 50

def extract_frames(video_path, frame_size=FRAME_SIZE):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = frame / 255.0  
        frames.append(frame)
    cap.release()
    return frames

def create_sequences(frames, sequence_length=SEQUENCE_LENGTH):
    if len(frames) < sequence_length:
        padding = [np.zeros_like(frames[0])] * (sequence_length - len(frames))
        frames.extend(padding)
    else:
        frames = frames[:sequence_length]
    return np.array(frames)

def load_dataset(dataset_path):
    X = []
    y = []
    class_names = sorted(os.listdir(dataset_path))
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            frames = extract_frames(video_path)
            sequence = create_sequences(frames)
            X.append(sequence)
            y.append(class_name)

    X = np.array(X)
    y = label_encoder.transform(y)  
    y = to_categorical(y, num_classes=NUM_CLASSES)  
    return X, y
    
print("Loading training data...")
X_train, y_train = load_dataset('dataset/train')
print("Loading validation data...")
X_val, y_val = load_dataset('dataset/val')
print("Loading test data...")
X_test, y_test = load_dataset('dataset/test')

def create_feature_extractor(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten()
    ])
    return model
input_shape = FRAME_SIZE + (3,)  
feature_extractor = create_feature_extractor(input_shape)

model = Sequential([
    TimeDistributed(feature_extractor, input_shape=(SEQUENCE_LENGTH, *input_shape)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

model.save('gesture_recognition_model.h5')
print("Model saved!")
