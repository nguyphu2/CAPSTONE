import time
import serial
import os
import csv

# ============================
# SERIAL CONFIG
# ============================

PORT = "COM6"
BAUD = 115200
sensor_ser = None

# ============================
# LOG FILE SETUP
# ============================

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

start_time_str = time.strftime("%m.%d.%y-%H.%M")
csv_file = os.path.join(log_dir, f"{start_time_str}.csv")

CSV_HEADER = [
    "time_s",
    "pir_l",
    "pir_r",
    "us_l",
    "us_m",
    "us_r",
    "yolo_class",
    "yolo_conf",
    "yolo_cam"
]

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

# ============================
# SERIAL INIT
# ============================

def serial_init():
    global sensor_ser
    try:
        sensor_ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[ARDUINO] Connected on {PORT}")
    except serial.SerialException as e:
        print(f"[ARDUINO] Connection failed: {e}")
        sensor_ser = None

# ============================
# MAIN LOOP (THREAD ENTRY)
# ============================

def loop(now, match_yolo, stop_event):
    """
    Arduino logging loop.
    Expects Arduino to send:
    PIR_L,PIR_R,US_L,US_M,US_R
    Example: 1,0,134,92,141
    """

    serial_init()
    print(f"[ARDUINO] Logging to {csv_file}")

    while not stop_event.is_set():

        # Attempt reconnect if disconnected
        if sensor_ser is None:
            serial_init()
            time.sleep(1)
            continue

        try:
            if not sensor_ser.in_waiting:
                time.sleep(0.01)
                continue

            line = sensor_ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            sensor_time = now()
            yolo_match = match_yolo(sensor_time)

            log_event(sensor_time, line, yolo_match)

        except serial.SerialException as e:
            print(f"[ARDUINO] Serial error: {e}")
            try:
                sensor_ser.close()
            except:
                pass
            sensor_ser = None

        except Exception as e:
            print(f"[ARDUINO] Unexpected error: {e}")

    # Clean shutdown
    if sensor_ser:
        sensor_ser.close()
    print("[ARDUINO] Shutdown complete")

# ============================
# CSV LOGGING
# ============================

def log_event(sensor_time, line, yolo_match):
    """
    Convert Arduino CSV payload into structured CSV row.
    """

    try:
        pir_l, pir_r, us_l, us_m, us_r = line.split(",")
    except ValueError:
        print(f"[LOG] Bad sensor payload: {line}")
        return

    if yolo_match:
        row = [
            f"{sensor_time:.3f}",
            pir_l,
            pir_r,
            us_l,
            us_m,
            us_r,
            yolo_match["class"],
            f"{yolo_match['conf']:.2f}",
            yolo_match["cam"]
        ]
    else:
        row = [
            f"{sensor_time:.3f}",
            pir_l,
            pir_r,
            us_l,
            us_m,
            us_r,
            "", "", ""
        ]

    try:
        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[LOG] Write error: {e}")
