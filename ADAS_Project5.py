import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PROGRAM 5
# Vehicle Detection using OpenCV DNN with SSD MobileNet
# =====================================================


# =====================================================
# 1. MODEL FILES
# =====================================================

CONFIG = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS = "frozen_inference_graph.pb"


# =====================================================
# 2. LOAD MODEL
# =====================================================

print("Loading SSD MobileNet model...")

net = cv2.dnn_DetectionModel(
    WEIGHTS,
    CONFIG
)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean(
    (127.5, 127.5, 127.5)
)
net.setInputSwapRB(True)

print("Model loaded successfully.")


# =====================================================
# 3. COCO CLASS NAMES
# =====================================================

classNames = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]


# =====================================================
# 4. READ VEHICLE IMAGE
# =====================================================

image = cv2.imread("vehicle.jpg")

if image is None:

    print("\nERROR: vehicle.jpg was not found.")

    print(
        "Put vehicle.jpg in the same folder "
        "as ADAS_Project5.py"
    )

    exit()


print("Vehicle image loaded successfully.")


# =====================================================
# 5. COPY IMAGE
# =====================================================

output = image.copy()


# =====================================================
# 6. DETECT OBJECTS
# =====================================================

classIds, scores, boxes = net.detect(
    image,
    confThreshold=0.45,
    nmsThreshold=0.40
)


# =====================================================
# 7. CONVERT RESULTS TO ARRAYS
# =====================================================

classIds = np.array(classIds).flatten()
scores = np.array(scores).flatten()


# =====================================================
# 8. VEHICLE CLASSES
# =====================================================

vehicle_names = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# =====================================================
# 9. DRAW VEHICLE BOUNDING BOXES
# =====================================================

vehicle_count = 0


for classId, score, box in zip(
    classIds,
    scores,
    boxes
):

    classId = int(classId)


    # Safety check
    if classId >= len(classNames):
        continue


    name = classNames[classId]


    # Only process vehicles
    if name in vehicle_names:

        vehicle_count += 1

        x, y, w, h = box


        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Display class name and confidence
        label = f"{name} {score:.2f}"


        cv2.putText(
            output,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


# =====================================================
# 10. DISPLAY VEHICLE COUNT
# =====================================================

cv2.putText(
    output,
    f"Vehicles detected: {vehicle_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    2
)


# =====================================================
# 11. CONVERT BGR TO RGB
# =====================================================

output_rgb = cv2.cvtColor(
    output,
    cv2.COLOR_BGR2RGB
)


# =====================================================
# 12. SHOW RESULT
# =====================================================

plt.figure(
    figsize=(14, 8)
)

plt.imshow(output_rgb)

plt.title(
    "Vehicle Detection using SSD MobileNet"
)

plt.axis("off")

plt.show()


print("\nVehicle detection completed.")
print(f"Vehicles detected: {vehicle_count}")