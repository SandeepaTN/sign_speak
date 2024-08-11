
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import pyttsx3


# Load the sign language recognition model
model = load_model('isl.h5')

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Define actions
actions = ['hello', 'help' ,'home', 'no', 'phone', 'thank you', 'there' ,'victory', 'water','yes']
 
# Function to perform Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])



    
# Create a ThreadPoolExecutor with one thread
executor = ThreadPoolExecutor(max_workers=1)

def speak_async(text):
    engine = pyttsx3.init()  # Create a new instance of the speech synthesis engine
    engine.say(text)
    engine.runAndWait()

def speak(text):
    # Submit the speak_async function to the ThreadPoolExecutor
    executor.submit(speak_async, text)

# Function to predict sign from video
def predict_sign_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            frames.append(keypoints)
            if len(frames) == 30:
                sequence = np.array(frames)
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sign = actions[np.argmax(res)]
                # Speak the predicted sign
                speak(sign)
                frames = []  # Reset frames for next sequence
                return sign
    
    cap.release()
    

    
# Define exampless

examples = [
        ['videos/qwe.mp4'],
        ['videos/asd.mp4']
        
    ]

    
    
# Create Gradio Interface
iface = gr.Interface(predict_sign_from_video,
                     inputs="video",
                     outputs="text", 
                     title="Sign Speak",
                     description="Upload a video and get the predicted sign spoken and displayed.",
                     examples=examples,
                     cache_examples=False)
iface.launch(share=True)

