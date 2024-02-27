import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Create a canvas that can fit the video feed
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to close the application
        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.quit)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

        self.update()

    def update(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()

        if ret:
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

            # Display the frame in the GUI
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Schedule the next update
            self.window.after(15, self.update)
        else:
            print("Error: Could not read frame.")

    def quit(self):
        self.cap.release()
        self.window.quit()

if __name__ == "__main__":
    # Create a window and pass it to the Application object
    root = tk.Tk()
    app = App(root, "Hand Gesture Recognition App")
    root.mainloop()
