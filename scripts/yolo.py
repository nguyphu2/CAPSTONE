import cv2  # type: ignore
import csv
import os
from datetime import datetime
from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8n.pt")

PIXEL_AREA_MIN = 35000
CONF_THRESHOLD = 0.7

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



#get fps from camera
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  
    



output_file_avi = "yolo_video(1).avi"

fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(output_file_avi, fourcc, fps * 0.9, (frame_width, frame_height))


detection_file = "example_file.csv"
with open(detection_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp1", "Frame Count1", "Area1","Number of boxes1", "Person_Detected1", "Person_Count1"])

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video with Timestamp', frame)
        results = model(frame, verbose=False)
        boxes = results[0].boxes


        person_count = 0
        annotated_frame = frame.copy()
        area_list = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            area = (x2 - x1) * (y2 - y1)
            area_list.append(area)
            
            if (cls_id == 0 and conf >= CONF_THRESHOLD and area >= PIXEL_AREA_MIN):
                person_count += 1
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                
        person_detected = 1 if person_count > 0 else 0 
        writer.writerow([timestamp, frame_count, area_list, len(boxes), person_detected, person_count])

        cv2.imshow("YOLOv8 Camera", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()



output_file_mp4 = "yolo_video(1).mp4"
os.system(f'ffmpeg -y -i "{output_file_avi}" -vcodec libx264 -crf 23 "{output_file_mp4}"')

print(f"Saved as {output_file_mp4}")
