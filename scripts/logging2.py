import time
import serial
import os
import csv

port = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44231313430351119231-if00'
baud = 115200
sensor_ser = None

# Sensor log setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file name with start date/time
start_time_str = time.strftime("%m.%d.%y-%H.%M")  # e.g., 11.06.25-14.05
log_file = os.path.join(log_dir, f"{start_time_str}.csv")

###############################################################
def loop(now, match_yolo, stop_event):
    """
    Main loop for Arduino sensor reading with YOLO fusion.
    
    Args:
        now: Function that returns current timestamp (relative to start)
        match_yolo: Function to find matching YOLO detection
        stop_event: threading.Event to signal shutdown
    """
    global sensor_ser
    
    # Initialize serial connection
    if not serial_init():
        print("[Arduino] Failed to initialize, exiting thread")
        stop_event.set()
        return
    
    print(f"[Arduino] Logging to: {log_file}")
    
    # Write CSV header
    try:
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "relative_time",
                "pir_right",
                "pir_middle",
                "pir_left",
                "ultrasonic",
                "rfid",
                "button",
                "yolo_match",
                "yolo_time",
                "yolo_total",
                "yolo_per_camera"
            ])
    except Exception as e:
        print(f"[Arduino] Failed to write CSV header: {e}")
    
    # Main loop
    while not stop_event.is_set():
        if sensor_ser is None:
            if not serial_init():
                time.sleep(1)  # Wait before retry
                continue
        
        try:
            # Check for data
            if not sensor_ser.in_waiting:
                time.sleep(0.01)  
                continue
        except serial.SerialException as e:
            print(f"[Arduino] Serial port error: {e}")
            close_serial()
            continue
        
        try:
            # Read sensor data
            line = sensor_ser.readline().decode('utf-8').strip()
            if not line:
                continue
            
            # Parse the Arduino message
            sensor_data = parse_arduino_message(line)
            if sensor_data is None:
                continue
            
            # Get timestamp and match with YOLO
            sensor_time = now()
            yolo_match = match_yolo(sensor_time)
            
            # Log the event with fusion data
            log_event(sensor_data, sensor_time, yolo_match)
            
        except UnicodeDecodeError as e:
            print(f"[Arduino] Decode error: {e}")
        except ValueError as e:
            print(f"[Arduino] Invalid sensor data: {line}")
        except Exception as e:
            print(f"[Arduino] Error reading sensor data: {e}")
    
    # Cleanup
    print("[Arduino] Stopping sensor logging...")
    close_serial()
    print("[Arduino] Sensor logging stopped")

###############################################################
def parse_arduino_message(line):
    """
    Parse Arduino message format: <TYPE>VALUE>
    Returns dict with sensor states or None if invalid
    """
    if not line.startswith('<') or not line.endswith('>'):
        return None
    
    # Remove markers
    content = line[1:-1]
    if len(content) < 1:
        return None
    
    msg_type = content[0]
    value = content[1:] if len(content) > 1 else ""
    
    sensor_data = {
        'pir_right': 0,
        'pir_middle': 0,
        'pir_left': 0,
        'ultrasonic': 0,
        'rfid': 0,
        'button': 0
    }
    
    # Parse based on message type
    if msg_type == 'R':  # Right PIR
        sensor_data['pir_right'] = int(value) if value.isdigit() else 0
    elif msg_type == 'M':  # Middle PIR
        sensor_data['pir_middle'] = int(value) if value.isdigit() else 0
    elif msg_type == 'L':  # Left PIR
        sensor_data['pir_left'] = int(value) if value.isdigit() else 0
    elif msg_type == 'U':  # Ultrasonic
        sensor_data['ultrasonic'] = int(value) if value.isdigit() else 0
    elif msg_type == 'F':  # RFID
        sensor_data['rfid'] = 1
    elif msg_type == 'B':  # Button
        sensor_data['button'] = int(value) if value.isdigit() else 0
    elif msg_type == 'P':  # Generic PIR (legacy)
        sensor_data['pir_middle'] = int(value) if value.isdigit() else 0
    else:
        print(f"[Arduino] Unknown message type: {msg_type}")
        return None
    
    return sensor_data

###############################################################
def serial_init():
    """
    Initialize serial connection to Arduino.
    Returns True on success, False on failure.
    """
    global sensor_ser
    try:
        sensor_ser = serial.Serial(port, baud, timeout=1)
        print(f"[Arduino] Connected to sensor Arduino on {port}")
        time.sleep(2)  # Give Arduino time to reset
        # Flush any startup noise
        sensor_ser.reset_input_buffer()
        return True
    except serial.SerialException as e:
        print(f"[Arduino] Failed to connect on {port}: {e}")
        sensor_ser = None
        return False

###############################################################
def close_serial():
    global sensor_ser
    if sensor_ser is not None:
        try:
            sensor_ser.close()
            print("[Arduino] Serial connection closed")
        except Exception as e:
            print(f"[Arduino] Error closing serial: {e}")
        finally:
            sensor_ser = None

###############################################################
def log_event(sensor_data, sensor_time, yolo_match):
    """
    Log sensor event with YOLO fusion data to CSV.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract YOLO match info
    if yolo_match:
        yolo_matched = "yes"
        yolo_time = f"{yolo_match['time']:.3f}"
        yolo_total = yolo_match['total']
        yolo_per_camera = str(yolo_match['per_camera'])
    else:
        yolo_matched = "no"
        yolo_time = ""
        yolo_total = ""
        yolo_per_camera = ""
    
    # Write to CSV
    try:
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                f"{sensor_time:.3f}",
                sensor_data['pir_right'],
                sensor_data['pir_middle'],
                sensor_data['pir_left'],
                sensor_data['ultrasonic'],
                sensor_data['rfid'],
                sensor_data['button'],
                yolo_matched,
                yolo_time,
                yolo_total,
                yolo_per_camera
            ])
    except Exception as e:
        print(f"[Arduino] Logging error: {e}")

###############################################################
# Keep old main() for standalone testing
def main():
    """Standalone mode for testing without YOLO"""
    if not serial_init():
        print("Failed to initialize, exiting")
        return
    
    print(f"Logging to: {log_file}")
    print("Press Ctrl+C to stop")
    
    # Write CSV header for standalone mode
    try:
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "relative_time",
                "pir_right",
                "pir_middle",
                "pir_left",
                "ultrasonic",
                "rfid",
                "button",
                "yolo_match",
                "yolo_time",
                "yolo_total",
                "yolo_per_camera"
            ])
    except Exception as e:
        print(f"Failed to write CSV header: {e}")
    
    try:
        while True:
            if sensor_ser is None:
                if not serial_init():
                    time.sleep(1)
                    continue
            
            try:
                if not sensor_ser.in_waiting:
                    time.sleep(0.01)
                    continue
                
                line = sensor_ser.readline().decode('utf-8').strip()
                if line:
                    sensor_data = parse_arduino_message(line)
                    if sensor_data:
                        log_event(sensor_data, time.time(), None)
                    
            except Exception as e:
                print(f"Error: {e}")
                close_serial()
                
    except KeyboardInterrupt:
        print("\nStopping...")
        close_serial()

###############################################################
if __name__ == "__main__":
    main()