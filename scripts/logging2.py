import time
import serial
import os

port = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44231313430351119231-if00'
baud = 115200
sensor_ser = None

# Sensor log setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file name with start date/time
start_time_str = time.strftime("%m.%d.%y-%H.%M")  # e.g., 11.06.25-14.05
log_file = os.path.join(log_dir, f"{start_time_str}.log")

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
    
    # Write header to log file
    try:
        with open(log_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Sensor logging started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Format: [Timestamp] Sensor_Data | YOLO_Match\n")
            f.write("=" * 80 + "\n")
    except Exception as e:
        print(f"[Arduino] Failed to write log header: {e}")
    
    # Main loop
    while not stop_event.is_set():
        if sensor_ser is None:
            if not serial_init():
                time.sleep(1)  # Wait before retry
                continue
        
        try:
            # Check for data
            if not sensor_ser.in_waiting:
                time.sleep(0.01)  # Small delay to prevent CPU spinning
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
            
            # Get timestamp and match with YOLO
            sensor_time = now()
            yolo_match = match_yolo(sensor_time)
            
            # Log the event with fusion data
            log_event(line, sensor_time, yolo_match)
            
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
def log_event(line, sensor_time, yolo_match):
    """
    Log sensor event with YOLO fusion data.
    """
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Format YOLO match info
    if yolo_match:
        yolo_info = (
            f"YOLO[t={yolo_match['time']:.3f}, "
            f"total={yolo_match['total']}, "
            f"cams={yolo_match['per_camera']}]"
        )
    else:
        yolo_info = "YOLO[no_match]"
    
    # Create log entry
    log_entry = f"{timestamp} t={sensor_time:.3f} | {line} | {yolo_info}\n"
    
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
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
                    log_event(line, time.time(), None)
                    
            except Exception as e:
                print(f"Error: {e}")
                close_serial()
                
    except KeyboardInterrupt:
        print("\nStopping...")
        close_serial()

###############################################################
if __name__ == "__main__":
    main()