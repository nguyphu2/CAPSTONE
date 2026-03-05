# Occupancy Detection System

A real-time occupancy detection system that fuses data from PIR sensors, ultrasonic sensors, and dual-camera YOLOv8 vision to classify room occupancy using a trained XGBoost model.

---

## System Overview

```
Arduino (PIR + Ultrasonic sensors)
        ↓
    
        
multi_thread.py  (runs Arduino and CV script)
        ↓                                                  ↓
log.py (parses and collects sensor data)  ←—→  non_stitch.py (YOLOv8, dual cameras)
        ↓
  Raw log files
        ↓
  raw.py  (log → CSV)
        ↓
  xg.py  (XGBoost training + W&B logging)
        ↓
  xgboost_best_model.json
```

---

## Project Structure

```
capstone/
├── multi_thread.py      # Thread orchestrator — runs YOLO + Arduino in parallel
├── non_stitch.py        # YOLOv8 dual-camera detection script
├── log.py               # Arduino serial reader + fused event logger
├── raw.py               # Parses .log files → structured CSV
├── xg.py                # XGBoost training pipeline with W&B integration
├── yolov8n.pt           # YOLOv8 nano weights (auto-downloaded by ultralytics)
├── logs/                # Raw .log files output by log.py
├── data/
│   ├── clean/           # Extracted CSV datasets (dataset 0–5)
│   └── results/         # Model outputs, plots, saved models
└── videos_outputs/      # Camera recordings from non_stitch.py
```

---

## Requirements

```bash
pip install opencv-python ultralytics xgboost scikit-learn imbalanced-learn wandb joblib matplotlib pandas numpy pyserial torch
```
---

## Hardware Setup

| Component | Description |
|---|---|
| Arduino | Reads PIR (left/right) and ultrasonic sensors (left, mid, right) |
| Camera 0 & 1 | USB cameras for YOLOv8 person detection |
| Serial port | Default `COM3` at `115200` baud — edit `log.py` to match your port |

Arduino serial output format (7 comma-separated values):
```
<sensor_time_ms>, <pir_left>, <pir_right>, <us_left>, <us_mid>, <us_right>, <flag>
```

---

## Usage

### 1. Data Collection

Run the main orchestrator to collect fused sensor + vision data:

```
python multi_thread.py
```

This starts two threads:
- **YOLO thread** (`non_stitch.py`) — opens both cameras, runs batched YOLOv8 inference, pushes detections to a shared buffer
- **Arduino thread** (`log.py`) — reads serial data, matches each reading to the nearest YOLO detection within a 50ms fusion window, writes to `logs/<timestamp>.log.`

Press `Q` in any camera window or `Ctrl+C` to stop.

#### Key config options in `non_stitch.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONF_THRESHOLD` | `0.7` | YOLO confidence threshold |
| `CAMERA_INDICES` | `[0, 1]` | Camera device indices |
| `FPS` | `30` | Target capture framerate |
| `IMG_SIZE` | `224` | YOLO inference resolution |
| `DRAW` | `True` | Enable bounding box overlay (disable for speed) |

#### Key config options in `log.py`:
| Parameter | Default | Description |
| `port` | `'COM3'` | Arduino serial port |
| `baud` | `115200` | Baud rate |

---

### 2. Log Extraction

Convert raw `.log` files to CSV for model training:

```
python raw.py
```

Edit the `LOG_FILE` and `CSV_FILE` paths at the top of the script before running.

Output CSV columns:

| Column | Description |
|---|---|
| `pir_left`, `pir_right` | PIR sensor readings (binary) |
| `us_left`, `us_mid`, `us_right` | Ultrasonic distances (cm) |
| `yolo_total` | Total people detected across both cameras |
| `yolo_cam0`, `yolo_cam1` | Per-camera person counts |

---

### 3. Model Training

```bash
python xg.py
```

Set `DATASET` (0–5) at the top of `xg.py` to select which cleaned CSV to use. Requires a [Weights & Biases](https://wandb.ai) account — comment out the `wandb` lines if not needed.

#### Key tuning parameters:

| Parameter | Default | Description |
|---|---|---|
| `DATASET` | `5` | Selects data directory |
| `TEST_SIZE` | `0.3` | Train/test split ratio |
| `DECISION_THRESHOLD` | `0.5` | Starting classification threshold (auto-optimized) |
| `USE_SMOTE` | `False` | Enable SMOTE oversampling |
| `USE_ADASYN` | `False` | Enable ADASYN oversampling |
| `EARLY_STOPPING_ROUNDS` | `20` | Stop training if no improvement |
| `OPTIMIZE_METRIC` | `"f1"` | Metric used for threshold optimization |

Outputs saved to `data/results/xgboost(N)/`:
- `xgboost_best_model.json` — trained model
- `scaler_best_model.pkl` — fitted StandardScaler
- `best_model_metadata.pkl` — features, threshold, F1 score
- `*_metrics.png` — performance bar charts
- `*_training_history.png` — loss/AUC curves

---

## Model Details

- **Task:** Binary classification — `Person Present` vs `No Person.`
- **Label:** `yolo_total > 0` (derived from YOLO ground truth)
- **Features:** `pir_left`, `pir_right`, `us_left`, `us_mid`, `us_right`
- **Algorithm:** XGBoost (`binary: logistic`) with dynamic `scale_pos_weight.`
- **Threshold:** Optimized per run on the validation set to maximize F1 for the positive class

---

## Notes

- The YOLO script (`non_stitch.py`) must fully initialize before the Arduino thread starts reading — this is handled automatically via a `threading.Event` (`yolo_ready`) in `main.py`.
- The 50ms fusion window (`FUSION_WINDOW` in `main.py`) can be tuned if sensor/camera timing drift is observed.
- Hardcoded Windows paths should be updated before running on a different machine.
