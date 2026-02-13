import time
import serial
import os
import xgboost as xgb
import numpy as np
import joblib

# =========================
# MODEL SETUP
# =========================
print("Loading XGBoost model...")
model = xgb.Booster()
model.load_model(r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost(2)\xgboost_best_model.json")

scaler = joblib.load(r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost(2)\scaler_best_model.pkl")
metadata = joblib.load(r"C:\Users\nguyphu2\Downloads\CAPSTONE\data\results\xgboost(2)\best_model_metadata.pkl")

THRESHOLD = metadata['threshold']
FEATURES = metadata['features']

print("✓ Model loaded successfully!")
print(f"  Features needed: {FEATURES}")
print(f"  Threshold: {THRESHOLD:.3f}\n")

def predict(sensor_values):
    """Predict person presence from sensor readings"""
    X = np.array([[sensor_values[feat] for feat in FEATURES]])
    X_scaled = scaler.transform(X)
    dmatrix = xgb.DMatrix(X_scaled)
    prob = model.predict(dmatrix)[0]
    pred = int(prob > THRESHOLD)
    return pred, prob

# =========================
# SERIAL SETUP
# =========================
port = 'COM3'
baud = 115200
sensor_ser = None
first_read = True

# =========================
# LOG FILE SETUP
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

start_time_str = time.strftime("%m.%d.%y-%H.%M")
log_file = os.path.join(log_dir, f"{start_time_str}.log")

# =========================
# SERIAL INIT
# =========================
def serial_init():
    global sensor_ser
    try:
        sensor_ser = serial.Serial(port, baud, timeout=1)
        print(f"[LOGGER] Connected to Arduino on {port}")
    except serial.SerialException as e:
        print(f"[LOGGER] FAILED to connect on {port}: {e}")
        sensor_ser = None

# =========================
# PARSE SENSOR LINE
# =========================
def parse_sensor_line(line):
    """
    Parse Arduino CSV format: timestamp, pir_left, pir_right, us_left, us_mid, us_right, unknown
    Example: "32343, 0, 0, 400.96, 479.20, 94.67, 1"
    """
    try:
        values = [v.strip() for v in line.split(',')]
        
        if len(values) < 7:
            print(f"[WARNING] Expected 7 values, got {len(values)}")
            return None
        
        return {
            'timestamp': float(values[0]),
            'pir_left': float(values[1]),
            'pir_right': float(values[2]),
            'us_left': float(values[3]),
            'us_mid': float(values[4]),
            'us_right': float(values[5]),
            'unknown': float(values[6])
        }
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Failed to parse sensor line: {line} | {e}")
        return None

# =========================
# SENSOR READ WITH PREDICTION
# =========================
def read_sensor_and_predict():
    global sensor_ser, first_read

    if sensor_ser is None:
        serial_init()
        if sensor_ser is None:
            return None, None, None

    try:
        line = sensor_ser.readline().decode("utf-8").strip()
        if not line:
            return None, None, None
    except serial.SerialException as e:
        print(f"[LOGGER] Serial error: {e}")
        sensor_ser = None
        return None, None, None

    if first_read:
        print("[LOGGER] FIRST SENSOR DATA RECEIVED")
        first_read = False

    print(f"[LOGGER] Received: {line}")

    # Parse sensor data
    sensor_data = parse_sensor_line(line)
    if sensor_data is None:
        return line, None, None

    # Make prediction
    try:
        # Extract only the features the model needs
        model_input = {feat: sensor_data[feat] for feat in FEATURES}
        prediction, probability = predict(model_input)
        
        status = "🚨 PERSON DETECTED" if prediction == 1 else "✓ No person"
        print(f"[PREDICTION] {status} (confidence: {probability:.2%})")
        
        return line, prediction, probability
    except KeyError as e:
        print(f"[ERROR] Missing required feature: {e}")
        print(f"[ERROR] Available: {sensor_data.keys()}, Needed: {FEATURES}")
        return line, None, None

# =========================
# FILE LOGGING WITH PREDICTIONS
# =========================
def log_event(line, prediction=None, probability=None, yolo_match=None):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    try:
        with open(log_file, "a") as f:
            log_line = f"{timestamp} {line}"
            
            if prediction is not None:
                pred_label = "PERSON" if prediction == 1 else "EMPTY"
                log_line += f" | ML: {pred_label} ({probability:.2%})"
            
            if yolo_match:
                log_line += f" | YOLO: {yolo_match}"
            
            log_line += "\n"
            
            f.write(log_line)
            f.flush()

        print(f"[LOGGER] Logged")

    except Exception as e:
        print(f"[LOGGER] LOGGING ERROR: {e}")

# =========================
# MAIN LOOP (THREADED MODE)
# =========================
def loop(now, match_yolo, stop_event):
    """Main loop for threaded execution with predictions"""
    global sensor_ser

    print("[LOGGER] loop() entered WITH ML PREDICTIONS")
    serial_init()

    print(f"[LOGGER] Logging to: {os.path.abspath(log_file)}")

    with open(log_file, "a") as f:
        f.write("=== LOGGER STARTED WITH ML PREDICTIONS ===\n")
        f.write(f"Model: {metadata['dataset']} | Features: {metadata['feature_set']}\n")
        f.write(f"F1 Score: {metadata['f1_score']:.4f} | Threshold: {THRESHOLD:.3f}\n")
        f.write("="*60 + "\n")
        f.flush()

    last_heartbeat = time.time()
    
    # Statistics
    total_predictions = 0
    person_detected_count = 0

    while not stop_event.is_set():
        line, prediction, probability = read_sensor_and_predict()
        
        if line:
            sensor_time = now()
            yolo_match = match_yolo(sensor_time)
            log_event(line, prediction, probability, yolo_match)
            
            if prediction is not None:
                total_predictions += 1
                if prediction == 1:
                    person_detected_count += 1
                    
                    # 🔴 TAKE ACTION WHEN PERSON DETECTED
                    # Add your actions here

        # 🫀 HEARTBEAT
        if time.time() - last_heartbeat >= 5.0:
            if total_predictions > 0:
                detection_rate = (person_detected_count / total_predictions) * 100
                print(f"[STATS] Predictions: {total_predictions} | Person: {person_detected_count} ({detection_rate:.1f}%)")
            last_heartbeat = time.time()

        time.sleep(0.2)

    if sensor_ser is not None:
        try:
            sensor_ser.close()
            print("[LOGGER] Serial port closed")
        except:
            pass

# =========================
# STANDALONE MODE
# =========================
def main():
    print("[LOGGER] Running in STANDALONE mode WITH ML PREDICTIONS")
    serial_init()
    print(f"[LOGGER] Logging to: {os.path.abspath(log_file)}")

    with open(log_file, "a") as f:
        f.write("=== LOGGER STARTED (STANDALONE WITH ML) ===\n")
        f.write(f"Model: {metadata['dataset']} | Features: {metadata['feature_set']}\n")
        f.write(f"F1 Score: {metadata['f1_score']:.4f} | Threshold: {THRESHOLD:.3f}\n")
        f.write("="*60 + "\n")
        f.flush()

    total_predictions = 0
    person_detected_count = 0
    last_heartbeat = time.time()

    try:
        while True:
            line, prediction, probability = read_sensor_and_predict()
            
            if line:
                log_event(line, prediction, probability)
                
                if prediction is not None:
                    total_predictions += 1
                    if prediction == 1:
                        person_detected_count += 1

            # Stats every 5 seconds
            if time.time() - last_heartbeat >= 5.0:
                if total_predictions > 0:
                    detection_rate = (person_detected_count / total_predictions) * 100
                    print(f"[STATS] Predictions: {total_predictions} | Person: {person_detected_count} ({detection_rate:.1f}%)")
                last_heartbeat = time.time()

            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n[LOGGER] Exiting...")
        print(f"[STATS] Final - Total: {total_predictions} | Person: {person_detected_count}")
        if sensor_ser is not None:
            sensor_ser.close()

# =========================
if __name__ == "__main__":
    main()
