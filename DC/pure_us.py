import re
import csv
import ast
import os

LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.21.26-19.53.log"

LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.16.26-15.42.log"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.23.26-11.33.log"

CSV_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean3\pure_us.csv"

os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Match timestamp, sensor data, and YOLO dict
pattern = re.compile(
    r"\[(?P<timestamp>.*?)\]\s*"
    r"(?P<data>[\d\.,\s]+)\s*\|\s*YOLO:\s*(?P<yolo>\{.*\})"
)

rows = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        timestamp = match.group("timestamp")
        sensor_values = [v.strip() for v in match.group("data").split(",")]

        # Expect exactly 7 sensor values
        if len(sensor_values) != 7:
            continue

        (
            ms,
            pir_left,
            pir_right,
            us_left,
            us_middle,
            us_right,
            unknown_flag
        ) = sensor_values

        # Parse YOLO dictionary
        try:
            yolo_dict = ast.literal_eval(match.group("yolo"))
            yolo_total = int(yolo_dict["total"])
            yolo_cam0 = int(yolo_dict["per_camera"][0])
            yolo_cam1 = int(yolo_dict["per_camera"][1])
        except Exception:
            continue

        rows.append([
            timestamp,
            int(ms),
            float(us_left),
            float(us_middle),
            float(us_right),
            yolo_total,
            yolo_cam0,
            yolo_cam1
        ])

# Write CSV
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "ms",
        "us_left",
        "us_middle",
        "us_right",
        "yolo_total",
        "yolo_cam0",
        "yolo_cam1"
    ])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {CSV_FILE}")
