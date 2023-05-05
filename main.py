import cv2
import tkinter as tk
from tkinter import filedialog

# Load input image
from findRotationAngle import findRotationAngle
from rotateImage import rotateImage

# create a Tkinter window
root = tk.Tk()
root.withdraw()

# show a file dialog to select an image file
#file_path = filedialog.askopenfilename()
file_path = r'C:\Users\stavr\Downloads\text1.png'

# read the selected image file
img = cv2.imread(file_path)

# Find rotation angle
angle = findRotationAngle(img)
print('Rotation angle:', angle)

# Rotate image
rotated_img = rotateImage(img, angle)

# Show original and rotated image
cv2.imshow('Original Image', img)
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
