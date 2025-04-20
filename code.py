import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from background_rmv import background_remove, extract_part, scale_to_reference_circle, show_overlay
#LOAD IMAGE
image_path = r"C:\Users\farha\OneDrive\Documents\Images\ref1.jpg"   
ref_path = r"C:\Users\farha\OneDrive\Documents\Images\ref1.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Circle Masked Out")

img, _ = background_remove(img,ref_path)
ref, _ , _ = extract_part(ref)

plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Circle Masked Out")

# Create a histogram of the image
plt.figure()
plt.hist(img.ravel(), bins=256, range=(0, 255))
plt.xlabel('Grayscale value')
plt.ylabel('Number of pixels')
plt.show()

ref = scale_to_reference_circle(ref, img)

plt.figure()
plt.imshow(ref, cmap='gray')
plt.title("Circle Masked Out")

plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Circle Masked Out")

show_overlay(img, ref, alpha=0.7)