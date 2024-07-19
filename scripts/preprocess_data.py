# preprocess_data.py
import cv2
import mediapipe as mp
import numpy as np
import os

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

def process_images(image_dir, save_file="data/processed/data.npz"):
    data = []
    labels = []
    expected_keypoints = 21 * 3  # 21 keypoints, each with x, y, z coordinates
    
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".png"):
            label = file_name.split('_')[0]
            frame = cv2.imread(os.path.join(image_dir, file_name))
            keypoints = extract_keypoints(frame)
            if keypoints.size == expected_keypoints:  # Ensure only frames with expected keypoints are included
                data.append(keypoints)
                labels.append(label)
    np.savez(save_file, data=np.array(data), labels=np.array(labels))

if __name__ == "__main__":
    process_images("data/raw")
