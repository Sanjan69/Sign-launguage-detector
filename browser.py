import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import pyttsx3
import speech_recognition as sr

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Define labels dictionary
labels_dict = {0: 'call me', 1: 'dislike', 2: 'you', 3: 'rock', 4: 'no', 5: 'yes', 6: 'i love you', 7: 'hello', 8: 'ok',
               9: 'peace', 10: 'high-five', 11: 'power', 12: 'point', 13: 'cool', 14: 'what do you want',
               15: 'im hungry', 16: 'i dont have any',17: 'come here', 18: 'A', 19: 'B', 20: 'C', 21: 'D', 22: 'E',
               23: 'F', 24: 'G', 25: 'H', 26: 'I', 27: 'J', 28: 'K', 29: 'L', 30: 'M', 31: 'N', 32: 'O', 33: 'P',
               34: 'Q', 35: 'R', 36: 'S', 37: 'T', 38: 'U', 39: 'V', 40: 'W', 41: 'X', 42:'Y', 43: 'Z'}

# Initialize MediaPipe Hands model and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

app = Flask(__name__)

def generate_frames():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe Hands
            results = hands.process(frame_rgb)

            # If hands are detected in the image, extract landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        # Normalize landmark coordinates and append to data_aux
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)

                        x_.append(landmark.x)
                        y_.append(landmark.y)

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10

                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

                # Ensure data_aux has the correct number of features
                if len(data_aux) != 42:
                    print("Error: Incorrect number of features extracted from hand landmarks.")
                else:
                    # Duplicate data to match the expected input size of the model
                    data_aux = np.array(data_aux * 2)

                    # Make prediction
                    prediction = model.predict([data_aux])
                    predicted_character = labels_dict[int(prediction[0])]
                    print(predicted_character)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                    # Speak the predicted character
                    engine.say(predicted_character)
                    engine.runAndWait()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')  # render the template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # return the response generator

if __name__ == "__main__":
    app.run(debug=True)
