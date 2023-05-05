import cv2
import numpy as np
import matplotlib as plt

from rotateImage import rotateImage


def findRotationAngle(image):
    # Convert image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayImage', grayImage)

    # Blur image to merge letter edges
    blurImage = cv2.blur(grayImage, (16, 16), 0)
    cv2.imshow('blurImage', blurImage)

    # Compute DFT of the blurImage
    discreteFourier = np.fft.fft2(blurImage)
    discreteFourierShift = np.fft.fftshift(discreteFourier)

    # Compute the magnitude spectrum of the DFT
    magnitudeSpectrum = 20 * np.log(np.abs(discreteFourierShift))

    # Find the row with the maximum value in the magnitude spectrum
    maxRow = np.argmax(magnitudeSpectrum, axis=0)[0]

    # Estimate the rotation angle based on the row with the maximum value
    angleEstimation = (maxRow - magnitudeSpectrum.shape[0] // 2) * 180 / magnitudeSpectrum.shape[0]
    angleEstimation = -angleEstimation  # Convert to counter-clockwise rotation

    # Define search range for rotation angle
    angleRange = np.arange(angleEstimation - 5, angleEstimation + 5, 0.5)

    # Initialize variables for best angle and corresponding projection value
    bestAngle = angleRange[0]
    maxVal = -1

    # Iterate through each angle in the search range
    for angle in angleRange:
        # Rotate image
        rotatedImage = rotateImage(image, angle)

        # Compute vertical projection of rotated image
        verticalProjection = np.sum(rotatedImage, axis=1)

        # Compute brightness changes in projection
        brightnessChanges = np.abs(np.diff(verticalProjection))

        # Compute sum of brightness changes
        projectionValue = np.sum(brightnessChanges)

        # Update best angle and corresponding projection value
        if projectionValue > maxVal:
            maxVal = projectionValue
            bestAngle = angle

    return bestAngle
