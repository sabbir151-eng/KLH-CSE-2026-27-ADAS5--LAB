import cv2
import numpy as np

# Load image from Downloads folder
image = cv2.imread(r"C:\Users\abdul\Downloads\DAY1.jpg")

# Check if image loaded
if image is None:
    print("Image not found!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate average brightness
brightness = np.mean(gray)

print("Ambient Brightness:", brightness)

# Headlight decision
if brightness < 80:
    print("Headlights ON")
else:
    print("Headlights OFF")

# Display image
cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()