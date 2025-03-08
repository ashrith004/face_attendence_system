import cv2
import pickle
import cv2.data
import numpy as np
import os

# Use OpenCV's default path for haar cascades
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
faces_data = []
i = 0
name = input("Enter Your Name :")

# Check if the classifier is loaded correctly
if facedetect.empty():
    print("Error: Could not load Haar cascade classifier.")
    exit()

# Open the webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (60, 60))  # Resize to 60x60 to match new dataset size

        # Append the face data only if it's the correct size and the condition meets
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        # Show the number of faces captured
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the frame with detected faces
    cv2.imshow("Frame", frame)

    # Wait for 'a' key press to exit the loop
    k = cv2.waitKey(1)
    if k == ord('a') or len(faces_data) == 100:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert faces_data list to a numpy array (100, 60, 60, 3)
faces_data = np.asarray(faces_data)

# Flatten faces_data to (100, 60*60*3)
faces_data = faces_data.reshape(faces_data.shape[0], -1)

# Ensure that the 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Handle names.pkl
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Handle faces_data.pkl
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    # Ensure compatibility of shapes
    faces = np.asarray(faces)  # Convert to numpy array if not already
    if faces.shape[1] != faces_data.shape[1]:
        print(f"Error: Face data dimensions do not match! Expected {faces.shape[1]}, but got {faces_data.shape[1]}.")
        print("Resetting faces_data.pkl to avoid inconsistencies.")
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
        exit()
    else:
        faces = np.append(faces, faces_data, axis=0)

    # Save updated faces data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print("Data saved successfully!")
