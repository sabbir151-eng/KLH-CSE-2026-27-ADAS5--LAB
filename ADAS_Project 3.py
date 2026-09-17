from ultralytics import YOLO
import cv2
import time


# =========================================================
# FORWARD COLLISION WARNING SYSTEM
# =========================================================

print("Starting Forward Collision Warning System...")


# =========================================================
# 1. LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully.")


# =========================================================
# 2. OPEN VIDEO
# =========================================================

video_path = "test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("\nERROR: test.mp4 could not be opened.")
    print("Make sure test.mp4 is in the same folder as ADAS_Project.py")
    input("\nPress Enter to exit...")
    exit()

print("Video opened successfully.")


# =========================================================
# 3. CAMERA PARAMETERS
# =========================================================

FOCAL_LENGTH = 700
KNOWN_WIDTH = 1.8


# =========================================================
# 4. TTC VARIABLES
# =========================================================

previous_distance = None
previous_time = None
smoothed_ttc = None


# =========================================================
# 5. DISTANCE ESTIMATION
# =========================================================

def estimate_distance(box_width):

    box_width = max(box_width, 1)

    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

    return distance


# =========================================================
# 6. MAIN VIDEO LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    current_time = time.time()


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model(frame, verbose=False)[0]


    best_distance = None
    best_box = None


    # =====================================================
    # FIND CLOSEST VEHICLE
    # =====================================================

    for box in results.boxes:

        class_id = int(box.cls[0])

        object_name = model.names[class_id]


        # Only detect vehicles
        if object_name not in [
            "car",
            "truck",
            "bus",
            "motorcycle"
        ]:
            continue


        # Bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # Calculate bounding-box width
        width = x2 - x1


        # Estimate distance
        distance = estimate_distance(width)


        # Select closest vehicle
        if (
            best_distance is None
            or distance < best_distance
        ):

            best_distance = distance

            best_box = (
                x1,
                y1,
                x2,
                y2
            )


    # =====================================================
    # TTC CALCULATION
    # =====================================================

    ttc = float("inf")


    if best_distance is not None:

        if (
            previous_distance is not None
            and previous_time is not None
        ):

            dt = max(
                current_time - previous_time,
                0.001
            )


            # Relative speed
            relative_speed = (
                previous_distance - best_distance
            ) / dt


            # Vehicle is getting closer
            if relative_speed > 0.1:

                ttc = (
                    best_distance
                    / relative_speed
                )


                # Smooth TTC
                if smoothed_ttc is None:

                    smoothed_ttc = ttc

                else:

                    smoothed_ttc = (
                        0.7 * smoothed_ttc
                        + 0.3 * ttc
                    )


        # Save current values
        previous_distance = best_distance
        previous_time = current_time


        # =================================================
        # DRAW VEHICLE BOX
        # =================================================

        if best_box is not None:

            x1, y1, x2, y2 = best_box


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )


            # Distance text
            cv2.putText(
                frame,
                f"Distance: {best_distance:.1f} m",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


    # =====================================================
    # DISPLAY TTC
    # =====================================================

    if smoothed_ttc is not None:

        displayed_ttc = smoothed_ttc

    else:

        displayed_ttc = float("inf")


    # =====================================================
    # WARNING SYSTEM
    # =====================================================

    if displayed_ttc < 2:

        warning = "CRITICAL: COLLISION WARNING!"

        warning_color = (
            0,
            0,
            255
        )


    elif displayed_ttc < 4:

        warning = "CAUTION: REDUCE SPEED"

        warning_color = (
            0,
            165,
            255
        )


    else:

        warning = "SAFE"

        warning_color = (
            0,
            255,
            0
        )


    # =====================================================
    # DISPLAY WARNING
    # =====================================================

    cv2.putText(
        frame,
        warning,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        warning_color,
        3
    )


    # =====================================================
    # DISPLAY TTC TEXT
    # =====================================================

    if displayed_ttc == float("inf"):

        ttc_text = "TTC: INF"

    else:

        ttc_text = (
            f"TTC: {displayed_ttc:.2f} s"
        )


    cv2.putText(
        frame,
        ttc_text,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================================
    # SHOW VIDEO
    # =====================================================

    cv2.imshow(
        "Forward Collision Warning",
        frame
    )


    # =====================================================
    # PRESS Q TO EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================================================
# 7. CLEAN UP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("Program stopped.")
