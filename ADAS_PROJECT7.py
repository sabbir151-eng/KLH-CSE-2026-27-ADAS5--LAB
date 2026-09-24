import cv2

image = cv2.imread("people.jpg")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

boxes, weights = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

for (x, y, w, h) in boxes:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imshow("Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()