# record_signs.py
import cv2
import os

def process_video(video_path, prompt, save_dir="data/raw"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video for: {prompt}. Total frames: {frame_count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_dir, f"{prompt}_{i}.png"), frame)

if __name__ == "__main__":
    video_file = "data/videos/example_video.mp4"  # Path to your video file
    prompt = "I love you"  # Corresponding prompt for the video
    process_video(video_file, prompt)
