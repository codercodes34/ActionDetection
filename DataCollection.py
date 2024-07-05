import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string

# Initialize video capture
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize hand detector with maximum one hand detection
detector = HandDetector(maxHands=1)

# Parameters for image processing
offset = 20
imgSize = 300
base_folder = "Data"

# Create the base folder if it doesn't exist
os.makedirs(base_folder, exist_ok=True)

while True:
    # Capture frame-by-frame
    success, img = cap.read()

    # Check if the frame was captured correctly
    if not success:
        print("Error: Failed to capture image.")
        break

    # Ensure the captured image is not empty
    if img is None or img.size == 0:
        print("Error: Captured empty frame.")
        continue

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region from the frame with some offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the cropping was successful
        if imgCrop.size == 0:
            print("Error: Failed to crop the image. Check the bounding box values.")
            continue

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize and place the cropped image on the white background
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped image and the white background image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame with hand detections
    cv2.imshow("Image", img)

    # Wait for key press and check if it is a letter key
    key = cv2.waitKey(1)
    if key != -1 and chr(key) in string.ascii_lowercase:
        letter = chr(key)
        folder = os.path.join(base_folder, letter.upper())
        os.makedirs(folder, exist_ok=True)
        counter = len(os.listdir(folder)) + 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image saved: {counter}")

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
