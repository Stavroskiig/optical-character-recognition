import cv2
import numpy as np


def rotateImage(image, angle):
    # Calculate image center
    center = tuple(np.array(image.shape[1::-1]) / 2)

    # Get rotation matrix
    rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions
    absoluteCos = abs(rotationMatrix[0, 0])
    absoluteSin = abs(rotationMatrix[0, 1])
    newWidth = int(image.shape[0] * absoluteSin + image.shape[1] * absoluteCos)
    newHeight = int(image.shape[0] * absoluteCos + image.shape[1] * absoluteSin)

    # Adjust rotation matrix to take into account translation
    rotationMatrix[0, 2] += newWidth / 2 - center[0]
    rotationMatrix[1, 2] += newHeight / 2 - center[1]

    # Rotate image
    rotatedImage = cv2.warpAffine(image, rotationMatrix, (newWidth, newHeight), borderValue=(255, 255, 255))

    return rotatedImage
