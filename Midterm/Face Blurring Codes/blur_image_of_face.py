# USAGE
# py -3 blur_face.py --image "image file"

# Import the necessary packages
import numpy as np
import argparse
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
    
# Construct the arguments and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="The path that is your image file to blur")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load a face detector model
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load an input image, copy it to display original image and grab dimensions of image
image = cv2.imread(args["image"])
original = image.copy()
(h, w) = image.shape[:2]

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the face detections
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the detection
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        face = image[startY:endY, startX:endX]

        face = blur_face(face, factor=3.0)

        # store the blurred face in the output image
        image[startY:endY, startX:endX] = face

# Display the original image and the output image with the blurred face side by side
output = np.hstack([original, image])
cv2.imshow("Output", output)
cv2.waitKey(0)