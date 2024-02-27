import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands model and drawing utilities
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = './data'

# Check if DATA_DIR exists
if not os.path.exists(DATA_DIR):
    print(f"Directory '{DATA_DIR}' not found. Please make sure the directory exists.")
    exit()

data = []
labels = []

for label_dir in os.listdir(DATA_DIR):
    for img_name in os.listdir(os.path.join(DATA_DIR, label_dir)):
        data_aux = []

        # Read image and convert to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, label_dir, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Hands
        results = hands.process(img_rgb)

        # If hands are detected in the image, extract landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.extend([x, y])  # Append x and y coordinates of each landmark

            # Append data and label to the respective lists
            data.append(data_aux)
            labels.append(label_dir)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
