# USAGE
# For Windows Users
# py -3 blur_video_of_face.py

# For Linux and Mac Users
# python blur_video_of_face.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def blur_face(image, factor=3.0):
    # Automatically determine the size of the blurring kernel
    # based on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)

    # Ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1

    # Ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1

    # Apply a Gaussian blur to the input image using our computed kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


# Load a face detector model
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("The video stream will be started...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and
    # resize this frame it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]

            # utilize the face blur method
            face = blur_face(face, factor=3.0)

            # store the blurred face in the output image
            frame[startY:endY, startX:endX] = face

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# To cleanup
cv2.destroyAllWindows()
vs.stop()
