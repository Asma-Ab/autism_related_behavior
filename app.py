import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tempfile

# Load the pre-trained model
model = load_model('C:/Users/abida/Desktop/AIM/ProjetCV/ssbd_classification/LRCN_Conv_LSTM_64_DEF.h5')

# Load class names
all_classes_names = os.listdir('C:/Users/abida/Desktop/AIM/ProjetCV/ssbd/')
CLASSES_LIST = all_classes_names

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 96
SEQUENCE_LENGTH = 20

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

def predict_action(video_path):
    frames = frames_extraction(video_path)
    frames = np.array(frames) * 1  # Normalize frames
    predictions = model.predict(np.expand_dims(frames, axis=0))
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASSES_LIST[predicted_class_index]
    return predicted_class

# Streamlit app
st.title("Autism Spectrum Related Action Recognition App")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Display uploaded video
    st.video(temp_video_path)

    # Make predictions
    predicted_class = predict_action(temp_video_path)

    # Display result
    st.success(f"Predicted Action: {predicted_class}")

    # Remove the temporary file
    os.remove(temp_video_path)
