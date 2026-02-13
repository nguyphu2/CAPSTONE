import re
import csv
import ast
from collections import deque

LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.21.26-19.53.log"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.16.26-15.42.log"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.23.26-11.33.log"

CSV_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean3\wma.csv"


WINDOW = 20

# Regex to match rows WITH YOLO
pattern = re.compile(
    r"\[(?P<timestamp>.*?)\]\s*"
    r"(?P<data>[\d\.,\s]+)\s*\|\s*YOLO:\s*(?P<yolo>\{.*\})"
)

# Rolling windows for ultrasonic sensors
us_left_q = deque(maxlen=WINDOW)
us_middle_q = deque(maxlen=WINDOW)
us_right_q = deque(maxlen=WINDOW)

def weighted_moving_average(values):
    """
    Weighted moving average where newer values have higher weight.
    """
    n = len(values)
    weights = range(1, n + 1)
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)

rows_clean = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        timestamp = match.group("timestamp")

        sensor_values = [v.strip() for v in match.group("data").split(",")]
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

        # Parse YOLO (kept raw)
        yolo_dict = ast.literal_eval(match.group("yolo"))

        us_left_f = float(us_left)
        us_middle_f = float(us_middle)
        us_right_f = float(us_right)

        # Update rolling windows
        us_left_q.append(us_left_f)
        us_middle_q.append(us_middle_f)
        us_right_q.append(us_right_f)

        # Apply weighted moving average
        us_left_smooth = weighted_moving_average(us_left_q)
        us_middle_smooth = weighted_moving_average(us_middle_q)
        us_right_smooth = weighted_moving_average(us_right_q)

        rows_clean.append([
            timestamp,
            int(ms),
            int(pir_left),
            int(pir_right),
            us_left_smooth,
            us_middle_smooth,
            us_right_smooth,
            int(unknown_flag),
            float(yolo_dict["time"]),
            int(yolo_dict["total"]),
            int(yolo_dict["per_camera"][0]),
            int(yolo_dict["per_camera"][1]),
        ])

# Write CSV
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "ms",
        "pir_left",
        "pir_right",
        "us_left",
        "us_middle",
        "us_right",
        "unknown_flag",
        "yolo_time",
        "yolo_total",
        "yolo_cam0",
        "yolo_cam1"
    ])
    writer.writerows(rows_clean)

print(f"Saved {len(rows_clean)} rows to {CSV_FILE}")
