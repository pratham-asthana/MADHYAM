import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* General styling */
    .stApp {
        background-image: url("https://i.gifer.com/7VE.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
    }
    /* Title styling */
    .stTitle {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #FF6F61;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        animation: fadeIn 2s ease-in-out;
        margin-top: 2rem;
    }
    /* Subtitle styling */
    .stSubtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #FFD166;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        animation: slideIn 1.5s ease-in-out;
    }
    /* Button styling */
    .stButton button {
        background-color: #06D6A0;
        color: white;
        font-size: 18px;
        padding: 12px 28px;
        border-radius: 25px;
        border: none;
        transition: background-color 0.3s ease, transform 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton button:hover {
        background-color: #05C493;
        transform: scale(1.05);
    }
    /* Video and image styling */
    .stVideo, .stImage {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: slideIn 1s ease-in-out;
    }
    /* Text styling */
    .stText {
        font-size: 1.2rem;
        color: #F8F9FA;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        animation: fadeIn 1.5s ease-in-out;
    }
    /* Radio button styling */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .stRadio label {
        color: #F8F9FA;
        font-size: 1.1rem;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# Load the trained model
model = load_model("gesture_recognition_model.h5")

# List of gesture labels (replace with your actual labels)
gesture_labels = ["Hello", "Thank You", "Yes", "No", "None"]

# Function to map class index to gesture label
def get_gesture_label(class_index):
    if 0 <= class_index < len(gesture_labels):
        return gesture_labels[class_index]
    return "Unknown"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to extract hand landmarks
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
    return landmarks

# Function to convert landmarks to an image
def landmarks_to_image(landmarks, image_size=(64, 64)):
    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    if landmarks:
        # Scale landmarks to fit the image
        landmarks = np.array(landmarks).reshape(-1, 2)
        landmarks = (landmarks * image_size).astype(int)
        # Draw landmarks on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

# Function to preprocess landmarks for the model
def preprocess_landmarks(landmarks_sequence, image_size=(64, 64)):
    # Convert each frame's landmarks to an image
    frames = [landmarks_to_image(landmarks, image_size) for landmarks in landmarks_sequence]
    # Stack frames into a sequence
    frames = np.array(frames)
    # Reshape to match the model's input shape (e.g., 30 frames, 64x64, 3 channels)
    frames = frames.reshape(1, 30, 64, 64, 3)
    return frames

# Function to classify gesture using the trained model
def classify_gesture(landmarks_sequence):
    if len(landmarks_sequence) == 30:  # Ensure we have enough frames
        # Preprocess the landmarks
        input_data = preprocess_landmarks(landmarks_sequence)
        # Predict the gesture
        prediction = model.predict(input_data)
        class_index = np.argmax(prediction)
        return get_gesture_label(class_index)  # Return the gesture label
    elif len(landmarks_sequence) == 0:
        return "None"  # No landmarks detected
    else:
        return "Collecting frames..."  # Not enough frames yet

# Streamlit app
st.markdown("<h1 class='stTitle'>MADHYAM</h1>", unsafe_allow_html=True)
st.markdown("""
    <h3 class='stSubtitle'>
        Translate Sign Language in Real-Time or Upload a Video
    </h3>
    """, unsafe_allow_html=True)

# Option selection
option = st.radio("Choose an option:", ("Real-Time Detection", "Upload Video", "Speech-to-Text"))

if option == "Real-Time Detection":
    st.markdown("<div class='stText'>Click the button below to start your webcam and detect sign language gestures.</div>", unsafe_allow_html=True)
    run = st.button("Start Webcam")

    if run:
        st.markdown("<div class='stText'>Running real-time detection...</div>", unsafe_allow_html=True)
        FRAME_WINDOW = st.image([], use_column_width=True)
        camera = cv2.VideoCapture(0)

        # Initialize a list to store landmarks sequences
        landmarks_sequence = []

        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Extract landmarks
            landmarks = extract_landmarks(frame)
            if landmarks:
                landmarks_sequence.append(landmarks)
                if len(landmarks_sequence) > 30:
                    landmarks_sequence.pop(0)  # Keep only the last 30 frames
            else:
                landmarks_sequence = []  # Reset if no landmarks are detected

            # Classify gesture
            gesture = classify_gesture(landmarks_sequence)

            # Display the frame with gesture text
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(frame)

        camera.release()

elif option == "Upload Video":
    st.markdown("<div class='stText'>Upload a video file for sign language transcription.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        st.markdown("<div class='stText'>Processing video...</div>", unsafe_allow_html=True)

        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the video
        cap = cv2.VideoCapture("temp_video.mp4")
        landmarks_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks
            landmarks = extract_landmarks(frame)
            if landmarks:
                landmarks_sequence.append(landmarks)
                if len(landmarks_sequence) > 30:
                    landmarks_sequence.pop(0)  # Keep only the last 30 frames
            else:
                landmarks_sequence = []  # Reset if no landmarks are detected

            # Classify gesture
            gesture = classify_gesture(landmarks_sequence)

            # Display the frame with gesture text
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(frame, caption="Processed Frame", use_column_width=True)

        cap.release()

elif option == "Speech-to-Text":
    st.markdown("<div class='stText'>Click the button below to record your voice and convert speech to text.</div>", unsafe_allow_html=True)
    record = st.button("Start Recording")
    if record:
        st.markdown("<div class='stText'>Recording... Speak now.</div>", unsafe_allow_html=True)
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise and set a longer phrase time limit
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, phrase_time_limit=10)  # Record for up to 10 seconds
                try:
                    text = recognizer.recognize_google(audio)
                    st.markdown(f"<div class='stText'>Transcribed Text: {text}</div>", unsafe_allow_html=True)
                except sr.UnknownValueError:
                    st.markdown("<div class='stText'>Could not understand audio.</div>", unsafe_allow_html=True)
                except sr.RequestError:
                    st.markdown("<div class='stText'>Could not request results from Google Speech-to-Text API.</div>", unsafe_allow_html=True)
        except AttributeError:
            st.markdown("<div class='stText'>Microphone access not available. Please check PyAudio installation.</div>", unsafe_allow_html=True)