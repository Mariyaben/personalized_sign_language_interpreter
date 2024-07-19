from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import uuid
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model and label encoder
model = tf.keras.models.load_model("models/sign_language_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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

def interpret_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints(frame)
        if keypoints.size == 21 * 3:  # Ensure correct number of keypoints
            keypoints = keypoints.reshape(1, 21, 3)  # Reshape for the model
            prediction = model.predict(keypoints)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            predictions.append(predicted_label[0])
    cap.release()
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predictions = interpret_video(file_path)
            os.remove(file_path)
            return render_template('result.html', predictions=predictions)
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
