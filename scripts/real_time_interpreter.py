# real_time_interpreter.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
    return np.array(keypoints).flatten()

# Load the trained model and label encoder
model = tf.keras.models.load_model("models/sign_language_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to interpret signs from a video file
def interpret_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints(frame)
        if keypoints.size == 21 * 3:  # Ensure correct number of keypoints
            keypoints = keypoints.reshape(1, 21, 3)  # Reshape for the model
            prediction = model.predict(keypoints)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            print(f"Predicted Sign: {predicted_label[0]}")
    cap.release()

if __name__ == "__main__":
    video_path = "data/videos/example_video.mp4"  # Replace with your video file path
    interpret_video(video_path)
