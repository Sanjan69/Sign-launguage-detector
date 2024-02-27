import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 44
dataset_size = 100


# Check for available cameras
num_cameras = 44
available_cameras = []
for i in range(num_cameras):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

if not available_cameras:
    print("Error: No cameras available. Exiting.")
    exit()

print("Available cameras:", available_cameras)

# Prompt the user to select a valid camera index
while True:
    try:
        cap_index = int(input("Enter the index of the camera you want to use: "))
        if cap_index in available_cameras:
            break
        else:
            print("Error: Invalid camera index. Please choose from the available cameras.")
    except ValueError:
        print("Error: Please enter a valid integer.")

cap = cv2.VideoCapture(cap_index)
if not cap.isOpened():
    print(f"Error: Unable to open camera {cap_index}. Exiting.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Exiting.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Exiting.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
