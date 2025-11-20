import cv2 # type: ignore
import csv
import os
from datetime import datetime
from ultralytics import YOLO # type: ignore
import numpy as np
import serial #type: ignore
import time


ser = serial.Serial("COM3", 115200, timeout = 1)
time.sleep(2)



def run():
    # Config
    FULLSCREEN = True
    MODEL_PATH = "yolov8n.pt"
    MIN_AREA = 35000
    CONF_THRESHOLD = 0.7
    OUTPUT_AVI = "stitched_yolo.avi"
    OUTPUT_MP4 = "stitched_yolo.mp4"
    OUTPUT_FILE = "stitched_detections.csv"
    INDV_FEEDS = True

    # Stitching configuration
    USE_STITCHING = True
    CAMERA_INDICES = [0, 1]
    STITCHER_MODE = cv2.Stitcher_SCANS  # SCANS works better for side-by-side cameras
    SKIP_STITCH_FRAMES = 1  # Only attempt stitching every N frames for performance
    USE_SIMPLE_CONCAT = False  # Set to True for guaranteed success (no stitching, just concatenate)
    AUTO_FALLBACK_THRESHOLD = 0.3  # If success rate < 30%, automatically switch to concatenation

    model = YOLO(MODEL_PATH)


    caps = [cv2.VideoCapture(i) for i in CAMERA_INDICES]

    # Set camera properties for better consistency
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Could not open camera {CAMERA_INDICES[i]}")
            exit()
        else:
            # Try to set consistent resolution and FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for consistency
            print(f"Camera {CAMERA_INDICES[i]} opened")

    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    # Adjust output dimensions
    if USE_STITCHING and len(caps) > 1:
        if USE_SIMPLE_CONCAT:
            output_width = frame_width * len(caps)
            output_height = frame_height
        else:
            # For stitching, estimate larger size
            output_width = int(frame_width * 1.8)
            output_height = frame_height
    else:
        output_width = frame_width
        output_height = frame_height

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_AVI, fourcc, fps, (output_width, output_height))

    # Initialize stitcher with better settings
    stitcher = None
    last_good_stitch = None
    stitch_failure_count = 0
    stitch_success_count = 0

    if USE_STITCHING and len(caps) > 1 and not USE_SIMPLE_CONCAT:
        stitcher = cv2.Stitcher_create(STITCHER_MODE)
        
        # Configure stitcher for better reliability
        try:
            # Adjust confidence threshold (lower = more lenient)
            stitcher.setRegistrationResol(0.6)  # Default is 0.6
            stitcher.setSeamEstimationResol(0.1)  # Default is 0.1
            stitcher.setCompositingResol(-1)  # Use full resolution
            stitcher.setPanoConfidenceThresh(0.5)  # Lower threshold (default is 1.0)
            print("Stitcher configured with optimized settings")
        except Exception as e:
            print(f"Could not configure stitcher settings: {e}")

    def simple_concat(frames):
        """Simple horizontal concatenation as fallback"""
        return np.hstack(frames)

    def convert_avi_to_mp4(input_path, output_path):
        """Fallback AVI → MP4 converter using OpenCV"""
        if not os.path.exists(input_path):
            print(f"Input file {input_path} not found")
            return
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Could not open {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_converter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Converting {input_path} → {output_path} ...")
        frame_count = 0
        
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_converter.write(frame)
            frame_count += 1

        cap.release()
        out_converter.release()
        print(f"Converted {frame_count} frames. Saved as {output_path}")

    # Open CSV file
    csv_file = None
    writer = None

    try:
        csv_file = open(OUTPUT_FILE, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Person_Count", "Area_List", "Stitch_Status"])
        
        frame_count = 0
        warmup_frames = 10  # Skip stitching for first N frames to let cameras stabilize
        
        print("=" * 60)
        
        
        
        # print("Waiting for Arduino READY...")
        # while True:
        #     line = ser.readline().decode().strip()
        #     if line == "READY":
        #         break

        # print("Arduino READY. Sending START...")

        # start_time = datetime.now()
        # ser.write(b"S")

        # while True:
        #     line = ser.readline().decode().strip()
        #     if line == "STARTED":
        #         print("Arduino started at:", datetime.now())
        #         break

        # print("SYNC COMPLETE. Starting YOLO detection loop now!")
        
        
        while True:
            frames = []
            
            # Read frames from all cameras
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Failed to read from camera {CAMERA_INDICES[i]}")
                    break
                frames.append(frame)
            
            # Check if we got all frames
            if len(frames) != len(caps):
                print("Lost camera connection")
                break
            
            # Display individual camera feeds
            if INDV_FEEDS:
                for i, frame in enumerate(frames):
                    cv2.imshow(f"Camera {CAMERA_INDICES[i]}", frame)
                
            stitch_status = "single_cam"
            
            # Process frame(s)
            if USE_SIMPLE_CONCAT and len(frames) > 1:
                # Simple concatenation - always works
                processed_frame = simple_concat(frames)
                stitch_status = "concatenated"
                
            elif stitcher is not None and len(frames) > 1:
                # Attempt stitching with optimizations
                should_stitch = (
                    frame_count >= warmup_frames and 
                    frame_count % SKIP_STITCH_FRAMES == 0
                )
                
                if should_stitch or last_good_stitch is None:
                    try:
                        status, stitched = stitcher.stitch(frames)
                        
                        if status == cv2.Stitcher_OK:
                            processed_frame = stitched
                            last_good_stitch = stitched.copy()
                            stitch_success_count += 1
                            stitch_status = "stitched"
                            
                            if stitch_success_count == 1:
                                print("✓ First successful stitch! Stitcher is now calibrated.")
                        else:
                            stitch_failure_count += 1
                            
                            # Use last good stitch or fallback
                            if last_good_stitch is not None:
                                processed_frame = last_good_stitch
                                stitch_status = "cached_stitch"
                            else:
                                processed_frame = simple_concat(frames)
                                stitch_status = f"concat_fallback_err{status}"
                            
                            # Only print error occasionally to avoid spam
                            if stitch_failure_count % 30 == 1:
                                error_msg = {
                                    1: "ERR_NEED_MORE_IMGS",
                                    2: "ERR_HOMOGRAPHY_EST_FAIL", 
                                    3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"
                                }.get(status, f"UNKNOWN_ERROR_{status}")
                                print(f"⚠ Stitching failures: {stitch_failure_count} ({error_msg})")
                                print("  → Check camera overlap and positioning")
                                
                    except Exception as e:
                        print(f"Stitching exception: {e}")
                        processed_frame = simple_concat(frames) if len(frames) > 1 else frames[0]
                        stitch_status = "exception"
                else:
                    # Reuse last good stitch for performance
                    if last_good_stitch is not None:
                        processed_frame = last_good_stitch
                        stitch_status = "cached_stitch"
                    else:
                        processed_frame = simple_concat(frames)
                        stitch_status = "concat_fallback"
            else:
                # Single camera
                processed_frame = frames[0]
                stitch_status = "single_cam"
            
            # Run YOLO detection
            results = model(processed_frame, imgsz = 320, verbose=False)
            boxes = results[0].boxes
            
            person_count = 0
            area_list = []
            annotated = processed_frame.copy()
            
            # Process detections
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                area = (x2 - x1) * (y2 - y1)
                
                if cls_id == 0 and conf >= CONF_THRESHOLD and area >= MIN_AREA:
                    person_count += 1
                    area_list.append(int(area))
                    
                    # Draw bounding boxes
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    
                    # Add confidence label
                    label = f"{conf:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            
            # Log to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow([timestamp, person_count, area_list, stitch_status])
            
            # Add overlay text with stitch info
            status_color = (0, 255, 0) if "stitch" in stitch_status else (255, 255, 0)
            cv2.putText(
                annotated,
                f"{timestamp} | People: {person_count} | {stitch_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            # Show stitch statistics
            if frame_count % 100 == 0 and stitch_success_count + stitch_failure_count > 0:
                success_rate = stitch_success_count / (stitch_success_count + stitch_failure_count) * 100
                cv2.putText(
                    annotated,
                    f"Stitch success: {success_rate:.1f}%",
                    (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    1,
                )
            
            # Show processed result
            cv2.imshow("YOLO Detection", annotated)
            
            if FULLSCREEN:
                cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
            
            else:
                cv2.setWindowProperty("YOLO Detection",
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL)
            # Write to video file
            if annotated.shape[1] != output_width or annotated.shape[0] != output_height:
                annotated = cv2.resize(annotated, (output_width, output_height))
            out.write(annotated)
            
            frame_count += 1
        
                
            key = cv2.waitKey(1) & 0xFF
            
            
            if key == ord(' '):  
                print("Space detected")
                INDV_FEEDS = not INDV_FEEDS
                if not INDV_FEEDS:
                    for i in CAMERA_INDICES:
                        cv2.destroyWindow(f"Camera {i}")
            # Exit on 'q' key
            elif key == ord("q"):
                print("Q detected")
                break
            
            elif key == ord("f"):
                print("f detected")
                FULLSCREEN = not FULLSCREEN
            
                   
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        
        for cap in caps:
            cap.release()
        
        out.release()
        
        if csv_file is not None:
            csv_file.close()
        
        cv2.destroyAllWindows()
        
        # Convert to MP4
        if os.path.exists(OUTPUT_AVI):
            convert_avi_to_mp4(OUTPUT_AVI, OUTPUT_MP4)
        else:
            print(f"No output file {OUTPUT_AVI} to convert")


if __name__ == "__main__":
    run()