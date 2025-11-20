import time
import serial
import os
import capstone.non_stitch as non_stitch

port = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44231313430351119231-if00'
baud = 115200
sensor_ser = None

# sensor log setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file name with start date/time
start_time_str = time.strftime("%m.%d.%y-%H.%M")  # e.g., 11.06.25-14.05
log_file = os.path.join(log_dir, f"{start_time_str}.log")

###############################################################
def main():
    serial_init()
    print(f"Logging to: {log_file}")
    while True:
        read_sensor()   
        time.sleep(0.2) 

###############################################################
def serial_init():
    global sensor_ser
    try:
        sensor_ser = serial.Serial(port, baud, timeout=1)
        print(f"Connected to sensor Arduino on {port}")
    except serial.SerialException as e:
        print(f"Failed to connect to sensor Arduino on {port}: {e}")
        sensor_ser = None

###############################################################
def read_sensor():
    global sensor_ser

    if sensor_ser is None:
        serial_init()
        if sensor_ser is None:
            return
    
    try: 
        if not sensor_ser.in_waiting:
            return
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
        try:
            sensor_ser.close()
        except: 
            pass
        sensor_ser = None
        return
    
    try:
        line = sensor_ser.readline().decode('utf-8').strip()
        if not line:
            return
        
        log_event(line)

    except ValueError:
        print("Invalid sensor data received:", line)
    except Exception as e:
        print("Error reading sensor data:", e)
    
###############################################################
def log_event(line):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    try:
        with open(log_file, "a") as f:
            f.write(f"{timestamp} {line}\n")
    except Exception as e:
        print(f"Logging error: {e}")

###############################################################
if __name__ == "__main__":
    main()
