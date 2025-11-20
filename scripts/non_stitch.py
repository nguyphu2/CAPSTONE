import cv2 #type: ignore
import csv
import os
from datetime import datetime
from ultralytics import YOLO #type: ignore
import numpy as np
import serial #type: ignore
import time


def run():
   
    
    # configus
    MODEL_PATH = "yolov8n.pt"
    MIN_AREA = 35000
    CONF_THRESHOLD = 0.7
    CAMERA_INDICES = [1, 2]   # Add more cameras here
    OUTPUT_DIR = "videos"
    OUTPUT_CSV = "multi_camera_detections.csv"

    SERIAL_PORT = "COM3"
    SERIAL_BAUD = 115200

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # print("Connecting to Arduino...")
    # ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    # time.sleep(2)
    # print("Arduino connected.")

  
    model = YOLO(MODEL_PATH)


    caps = []
    writers = []

    for cam_index in CAMERA_INDICES:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Could not open camera {cam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        caps.append(cap)

        video_path = os.path.join(OUTPUT_DIR, f"camera_{cam_index}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
        writers.append(writer)

        print(f"Camera {cam_index} initialized → saving to {video_path}")

    # -----------------------------
    # Initialize CSV
    # -----------------------------
    csv_file = open(OUTPUT_CSV, "w", newline="")
    writer_csv = csv.writer(csv_file)

    header = ["Timestamp"]
    for i in range(len(CAMERA_INDICES)):
        header.append(f"Cam{i}_Count")
        header.append(f"Cam{i}_Detection")
    header.append("Total_Count")
    writer_csv.writerow(header)

    print("=" * 60)
    print("Running NON-STITCHED multi-camera YOLO...")
    print("=" * 60)

    try:
        while True:

            per_camera_counts = []
            per_camera_detection = []
            for idx, (cam_index, cap) in enumerate(zip(CAMERA_INDICES, caps)):
                ret, frame = cap.read()
                if not ret:
                    print(f"Camera {cam_index} failed to read frame.")
                    per_camera_counts.append(0)
                    continue

                # Run YOLO
                results = model(frame, imgsz=320, verbose=False)
                boxes = results[0].boxes

                person_count = 0
                annotated = frame.copy()

                # Process detections
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    area = (x2 - x1) * (y2 - y1)

                    if cls_id == 0 and conf >= CONF_THRESHOLD and area >= MIN_AREA:
                        person_count += 1

                        cv2.rectangle(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            annotated,
                            f"{conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                per_camera_counts.append(person_count)
                per_camera_detection.append(person_count>0)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                cv2.putText(
                    annotated,
                    timestamp,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(f"Camera {cam_index}", annotated)

                writers[idx].write(annotated)

           
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            total_people = sum(per_camera_counts)
            row = [timestamp]

            for count, detected in zip(per_camera_counts, per_camera_detection):
                row.append(count)
                row.append(detected)

            row.append(total_people)

            writer_csv.writerow(row)


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Q pressed → Exiting...")
                break

    finally:
        print("Cleaning up...")

        for cap in caps:
            cap.release()
        for w in writers:
            w.release()

        csv_file.close()
        cv2.destroyAllWindows()
        print("Videos saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
