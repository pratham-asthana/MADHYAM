import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os
import pandas as pd

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

SEQUENCE_LENGTH = 30  
FRAME_SIZE = (64, 64) 
NUM_LANDMARKS = 21 * 3 + 33 * 3 
CLASS_NAMES = sorted(os.listdir('dataset/train'))  

model = load_model('gesture_recognition_model.h5')


def extract_landmarks(frame):
    landmarks = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

    
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

    
    if len(landmarks) < NUM_LANDMARKS:
        landmarks.extend([0] * (NUM_LANDMARKS - len(landmarks)))

    return np.array(landmarks[:NUM_LANDMARKS])  


def landmarks_to_image(landmarks, image_size=FRAME_SIZE):
    
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    
    landmarks = landmarks.reshape(-1, 3)

    
    landmarks[:, 0] = (landmarks[:, 0] * image_size[1]).astype(int)  
    landmarks[:, 1] = (landmarks[:, 1] * image_size[0]).astype(int)  

    
    for landmark in landmarks:
        x, y, _ = landmark
        x, y = int(x), int(y)  
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:  
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1) 

    return image


cap = cv2.VideoCapture(0)  


frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    landmarks = extract_landmarks(frame)
    landmark_image = landmarks_to_image(landmarks)

    frame_buffer.append(landmark_image)
    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        sequence = np.array(frame_buffer)
        sequence = sequence / 255.0  
        sequence = np.expand_dims(sequence, axis=0)  

        # Verify the sequence shape matches the model's input shape
        if sequence.shape[1:] != model.input_shape[1:]:
            raise ValueError(f"Input shape mismatch. Expected {model.input_shape[1:]}, but got {sequence.shape[1:]}")

        # Make a prediction
        prediction = model.predict(sequence)
        predicted_class = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class]

        # Display the predicted class on the frame
        cv2.putText(frame, f'Prediction: {predicted_class_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    cv2.imshow('Real-Time Gesture Recognition', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()