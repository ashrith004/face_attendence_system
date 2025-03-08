import cv2
import pickle
import numpy as np
import csv
import os
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Start video capture
video = cv2.VideoCapture(0)
# Load the face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

# Load the labels and faces data used to train the classifier
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Print the shape of the training data for debugging
print(f"Shape of FACES: {np.array(FACES).shape}")

# Adjust lengths to match each other
if len(LABELS) > len(FACES):
    print(f"Labels are more than faces. Trimming LABELS to match faces count.")
    LABELS = LABELS[:len(FACES)]  # Trim LABELS to match FACES length
elif len(FACES) > len(LABELS):
    print(f"Faces are more than labels. Trimming FACES to match labels count.")
    FACES = FACES[:len(LABELS)]  # Trim FACES to match LABELS length

# Train KNN classifiers
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME  ', 'TIME']
# Define the expected size from the training images (60x60 with 3 channels)
resize_width = 60  # The width of the image (same as training data)
resize_height = 60  # The height of the image (same as training data)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    # Convert the frame to grayscale (ensure consistency with training)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        crop_img = frame[y:y + h, x:x + w]
        # Resize face to 60x60 for both training and testing consistency
        resized_img = cv2.resize(crop_img, (resize_width, resize_height))  # Ensure 60x60 size
       # Check if the image is RGB (3 channels)
        if resized_img.ndim == 3:  
            resized_img = resized_img.flatten().reshape(1, -1)  # Flatten and reshape to match the model input
        elif resized_img.ndim == 2:  # If the image is grayscale (1 channel)
            resized_img = resized_img.flatten().reshape(1, -1)  # Flatten and reshape to match the model input
        # Predict label
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]
        
        # Display the predicted label on the frame
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the frame with detected faces
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('O'):
        speak("Attendance Taken successfully")
        time.sleep(1)
        if exist:  # type: ignore
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile: # type: ignore
                writer = csv.writer(csvfile)
                writer.writerow(attendance) # type: ignore
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile: # type: ignore
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance) # type: ignore
            csvfile.close()
    # Wait for 'a' key press to exit the loop
    if k == ord('a'):
        break
# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
