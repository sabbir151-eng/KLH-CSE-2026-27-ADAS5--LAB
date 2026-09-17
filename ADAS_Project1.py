# Forward Collision Warning System
# Uses YOLO object detection and OpenCV video processing

from ultralytics import YOLO
import cv2
import time


# --------------------------------------------------
# 1. Load YOLO model
# --------------------------------------------------
print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully.")


# --------------------------------------------------
# 2. Open input video
# --------------------------------------------------
video_path = "test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open test.mp4")
    print("Make sure test.mp4 is in the same folder as ADAS.py")
    exit()

print("Video opened successfully.")


# --------------------------------------------------
# 3. Camera and vehicle parameters
# --------------------------------------------------
FOCAL_LENGTH = 700
KNOWN_WIDTH = 1.8


# --------------------------------------------------
# 4. Variables for TTC calculation
# --------------------------------------------------
previous_distance = None
previous_time = None
smoothed_ttc = None


# --------------------------------------------------
# 5. Distance estimation function
# --------------------------------------------------
def estimate_distance(box_width):
    box_width = max(box_width, 1)

    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

    return distance


# --------------------------------------------------
# 6. Process video frame by frame
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    current_time = time.time()

    # Run YOLO detection
    results = model(frame, verbose=False)[0]

    best_distance = None
    best_box = None


    # --------------------------------------------------
    # 7. Find the closest vehicle
    # --------------------------------------------------
    for box in results.boxes:

        class_id = int(box.cls[0])

        name = model.names[class_id]

        # Only consider vehicles
        if name not in ["car", "truck", "bus", "motorcycle"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1

        distance = estimate_distance(width)


        # Select closest vehicle
        if best_distance is None or distance < best_distance:

            best_distance = distance

            best_box = (x1, y1, x2, y2)


    # --------------------------------------------------
    # 8. Calculate Time To Collision (TTC)
    # --------------------------------------------------
    ttc = float("inf")

    if best_distance is not None:

        if previous_distance is not None and previous_time is not None:

            dt = max(current_time - previous_time, 1e-3)

            # Positive value means distance is decreasing
            relative_speed = (
                previous_distance - best_distance
            ) / dt


            if relative_speed > 0.1:

                ttc = best_distance / relative_speed


                # Temporal smoothing
                if smoothed_ttc is None:

                    smoothed_ttc = ttc

                else:

                    smoothed_ttc = (
                        0.7 * smoothed_ttc
                        + 0.3 * ttc
                    )


        previous_distance = best_distance
        previous_time = current_time


        # --------------------------------------------------
        # 9. Draw vehicle bounding box
        # --------------------------------------------------
        if best_box is not None:

            x1, y1, x2, y2 = best_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )


            # Display estimated distance
            cv2.putText(
                frame,
                f"Distance: {best_distance:.1f} m",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


    # --------------------------------------------------
    # 10. Determine warning level
    # --------------------------------------------------
    displayed_ttc = (
        smoothed_ttc
        if smoothed_ttc is not None
        else float("inf")
    )


    if displayed_ttc < 2:

        warning = "CRITICAL: COLLISION WARNING!"

        warning_color = (0, 0, 255)


    elif displayed_ttc < 4:

        warning = "CAUTION: REDUCE SPEED"

        warning_color = (0, 165, 255)


    else:

        warning = "SAFE"

        warning_color = (0, 255, 0)


    # --------------------------------------------------
    # 11. Display warning
    # --------------------------------------------------
    cv2.putText(
        frame,
        warning,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        warning_color,
        3
    )


    # --------------------------------------------------
    # 12. Display TTC
    # --------------------------------------------------
    if displayed_ttc == float("inf"):

        ttc_text = "TTC: INF"

    else:

        ttc_text = f"TTC: {displayed_ttc:.2f} s"


    cv2.putText(
        frame,
        ttc_text,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------
    # 13. Show video
    # --------------------------------------------------
    cv2.imshow(
        "Forward Collision Warning",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------------------
# 14. Release resources
# --------------------------------------------------
cap.release()

cv2.destroyAllWindows()

print("Program stopped.")