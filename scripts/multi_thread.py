import threading
import capstone.non_stitch as non_stitch  
import logging  

def start_yolo():
    print("Starting YOLO thread...")
    non_stitch.run()

def start_arduino():
    print("Starting Arduino logging thread...")
    logging.loop()

if __name__ == "__main__":
    t1 = threading.Thread(target=start_yolo, daemon=True)
    t2 = threading.Thread(target=start_arduino, daemon=True)

    t1.start()
    t2.start()

    print("Threads started. Press Ctrl+C to exit.")

    t1.join()
    t2.join()

