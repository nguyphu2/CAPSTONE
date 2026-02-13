import re
import csv
import ast
import statistics

LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.21.26-19.53.log"
CSV_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean\us_capped.csv"
LOG_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\logs\01.16.26-15.42.log"
CSV_FILE = r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\clean2\us_capped.csv"


# Regex to match rows WITH YOLO
pattern = re.compile(
    r"\[(?P<timestamp>.*?)\]\s*"
    r"(?P<data>[\d\.,\s]+)\s*\|\s*YOLO:\s*(?P<yolo>\{.*\})"
)

rows_raw = []

# Store ultrasonic values for statistics
us_left_vals = []
us_middle_vals = []
us_right_vals = []

# -------------------------------
# First pass: parse + collect stats
# -------------------------------
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

        yolo_dict = ast.literal_eval(match.group("yolo"))

        us_left_f = float(us_left)
        us_middle_f = float(us_middle)
        us_right_f = float(us_right)

        us_left_vals.append(us_left_f)
        us_middle_vals.append(us_middle_f)
        us_right_vals.append(us_right_f)

        rows_raw.append({
            "timestamp": timestamp,
            "ms": int(ms),
            "pir_left": int(pir_left),
            "pir_right": int(pir_right),
            "us_left": us_left_f,
            "us_middle": us_middle_f,
            "us_right": us_right_f,
            "unknown_flag": int(unknown_flag),
            "yolo_time": float(yolo_dict["time"]),
            "yolo_total": int(yolo_dict["total"]),
            "cam0": int(yolo_dict["per_camera"][0]),
            "cam1": int(yolo_dict["per_camera"][1]),
        })

# -------------------------------
# Compute caps (mean + 1 std)
# -------------------------------
left_cap = statistics.mean(us_left_vals) + statistics.stdev(us_left_vals)
middle_cap = statistics.mean(us_middle_vals) + statistics.stdev(us_middle_vals)
right_cap = statistics.mean(us_right_vals) + statistics.stdev(us_right_vals)

# -------------------------------
# Second pass: apply YOLO-aware caps
# -------------------------------
rows_clean = []

for r in rows_raw:

    us_left_clean = r["us_left"]
    us_middle_clean = r["us_middle"]
    us_right_clean = r["us_right"]

    if r["yolo_total"] > 0:

        # cam0 → left + middle
        if r["cam0"] > 0:
            us_left_clean = min(us_left_clean, left_cap)

        # cam1 → right + middle
        if r["cam1"] > 0:
            us_right_clean = min(us_right_clean, right_cap)

        # middle sensor uses the lower applicable cap
        middle_caps = []
        if r["cam0"] > 0:
            middle_caps.append(middle_cap)
        if r["cam1"] > 0:
            middle_caps.append(middle_cap)

        if middle_caps:
            us_middle_clean = min(us_middle_clean, min(middle_caps))

    rows_clean.append([
        r["timestamp"],
        r["ms"],
        r["pir_left"],
        r["pir_right"],
        us_left_clean,
        us_middle_clean,
        us_right_clean,
        r["unknown_flag"],
        r["yolo_time"],
        r["yolo_total"],
        r["cam0"],
        r["cam1"],
    ])

# -------------------------------
# Write CSV
# -------------------------------
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
