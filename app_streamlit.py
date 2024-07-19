import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model
from pathlib import Path
import os

# Define file paths
MODEL_PATH = 'models/sign_language_model.h5'
DATA_DIR = 'data/raw'
TEMP_VIDEO_PATH = 'D:\personalized_sign_language_interpreter\data\videos\example_video.mp4'

# Load the trained model
model = load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.title("Sign Language Interpreter")

st.write("Upload a video to sign the text: I LOVE YOU")

# File uploader widget
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save the uploaded file
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.read())
    
    # Display the video
    st.video(TEMP_VIDEO_PATH)

    # Function to preprocess video and extract frames
    def preprocess_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [landmark for lm in hand_landmarks.landmark]
                    frames.append(landmarks)
        cap.release()
        return frames

    # Function to predict sign
    def predict_sign(frames):
        frames_array = np.array(frames)
        frames_array = np.expand_dims(frames_array, axis=0)  # Add batch dimension
        predictions = model.predict(frames_array)
        return np.argmax(predictions, axis=1)

    # Process and predict the sign
    frames = preprocess_video(TEMP_VIDEO_PATH)
    if frames:
        prediction = predict_sign(frames)
        # Map prediction to sign language text (example)
        signs = ["Hello", "I love you", "Thank you"]  # Update with your signs
        predicted_sign = signs[prediction[0]]
        st.write(f"Predicted sign: {predicted_sign}")
    else:
        st.write("Memory updated for prompt I LOVE YOU.")
