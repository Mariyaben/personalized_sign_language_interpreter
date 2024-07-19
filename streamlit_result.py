import streamlit as st

# Set the title of the Streamlit app
st.title("Sign Language Interpreter")

# Prompt the user to upload a video
st.header("Upload your sign video to get text interpretation")

# Create a file uploader widget for video files
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

# If a file is uploaded, process the file (dummy processing for this example)
if uploaded_file is not None:
    st.video(uploaded_file)
    # Here you would add the code to process the video and get the interpretation
    # For this example, we'll just display a fixed interpretation
    st.subheader("TEXT interpretation: I love you")
