import re
import csv
import ast

LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.21.26-19.53.log"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.16.26-15.42.log"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.23.26-11.33.log"

CSV_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean3\raw.csv"


pattern = re.compile(
    r"\[(?P<timestamp>.*?)\]\s*"
    r"(?P<data>[\d\.,\s]+)\s*\|\s*YOLO:\s*(?P<yolo>\{.*\})"
)

rows = []

# -------------------------
# 1. Parse log
# -------------------------
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
            sensor_time_ms,
            pir_left,
            pir_right,
            us_left,
            us_middle,
            us_right,
            unknown_flag
        ) = sensor_values

        yolo = ast.literal_eval(match.group("yolo"))

        rows.append({
            "time": timestamp,
            "sensor_time_ms": int(sensor_time_ms),
            "pir_left": int(pir_left),
            "pir_right": int(pir_right),
            "us_left": float(us_left),
            "us_middle": float(us_middle),
            "us_right": float(us_right),
            "unknown_flag": int(unknown_flag),
            "yolo_total": int(yolo["total"]),
            "yolo_cam0": int(yolo["per_camera"][0]),
            "yolo_cam1": int(yolo["per_camera"][1]),
        })

# -------------------------
# 2. Sort by time
# -------------------------
rows.sort(key=lambda r: r["sensor_time_ms"])

# -------------------------
# 3. Write CSV
# -------------------------
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time",
        "sensor_time_ms",
        "pir_left",
        "pir_right",
        "us_left",
        "us_middle",
        "us_right",
        "unknown_flag",
        "yolo_total",
        "yolo_cam0",
        "yolo_cam1",
    ])

    for r in rows:
        writer.writerow([
            r["time"],
            r["sensor_time_ms"],
            r["pir_left"],
            r["pir_right"],
            r["us_left"],
            r["us_middle"],
            r["us_right"],
            r["unknown_flag"],
            r["yolo_total"],
            r["yolo_cam0"],
            r["yolo_cam1"],
        ])

print(f"Extracted {len(rows)} rows to {CSV_FILE}")