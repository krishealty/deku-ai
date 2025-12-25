# Urban Traffic Intelligence System

A GPU-accelerated, end-to-end computer vision system for real-time traffic analysis and monitoring.

## Features

- **Real-time Multi-class Detection**: Vehicles (car, bus, truck, motorcycle, bicycle) and pedestrians using YOLOv8
- **Multi-Object Tracking**: Persistent ID assignment using DeepSORT/ByteTrack
- **Advanced Analytics**: 
  - Vehicle and pedestrian density
  - Flow rate measurement
  - Speed estimation
  - Crowd build-up and loitering detection
  - Traffic heatmaps
- **Anomaly Detection**:
  - Sudden vehicle stoppages
  - Potential collisions
  - Wrong-direction movement
  - Abnormal crowd surges
- **Dual Control Mechanisms**:
  - Hand gesture recognition (MediaPipe Hands)
  - Traditional dashboard UI controls
- **Interactive Dashboard**: Real-time metrics, graphs, heatmaps, and alerts

## Installation

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration, ensure you have CUDA installed and PyTorch with CUDA support. The system will automatically fall back to CPU if CUDA is not available.

## Usage

### Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Or use the helper script:
```bash
python run_streamlit.py
```

### OpenCV Display (Alternative)

Modify `main.py` to use `run_opencv_display()` method for a simple OpenCV window display.

## Configuration

Edit `config.yaml` to customize:
- Model type (yolov8l or yolov8x)
- Detection thresholds
- Tracking parameters (max_age, min_hits, IoU threshold)
- Analytics settings (grid size, speed calibration)
- Flow line definitions
- Anomaly detection thresholds
- Gesture control mappings
- Input source (webcam or video file)

## Features Overview

### Detection & Tracking
- **YOLOv8 Detection**: GPU-accelerated multi-class object detection
- **DeepSORT Tracking**: Persistent ID assignment with Kalman filtering
- **Real-time Processing**: Optimized for live video streams

### Analytics
- **Density Analysis**: Grid-based vehicle and pedestrian density
- **Flow Rate**: Virtual line crossing detection and counting
- **Speed Estimation**: Velocity calculation with calibration support
- **Loitering Detection**: Identify objects staying in small areas
- **Heatmaps**: Temporal traffic pattern visualization

### Anomaly Detection
- **Sudden Stoppages**: Detect abrupt vehicle stops
- **Collision Detection**: Identify potential collisions from overlaps and velocity drops
- **Wrong Direction**: Flag objects moving against expected flow
- **Crowd Surges**: Detect abnormal pedestrian density increases

### Control Mechanisms
- **Gesture Control**: MediaPipe Hands for finger-count-based mode switching
  - 1 finger: Vehicle density view
  - 2 fingers: Pedestrian density view
  - 3 fingers: Heatmap overlay
  - 4 fingers: Incident detection mode
  - 5 fingers: Full analytics view
- **Dashboard Controls**: Traditional UI with buttons, sliders, and toggles

### Visualization
- Bounding boxes with track IDs
- Trajectory paths
- Real-time metrics overlay
- Heatmap overlays
- Anomaly alert markers
- Flow line indicators

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Webcam or video file input
- 4GB+ RAM recommended
- 2GB+ VRAM for GPU acceleration

## Project Structure

```
deku/
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── main.py                  # Main application logic
├── app.py                   # Streamlit entry point
├── run_streamlit.py         # Helper script to run Streamlit
├── src/
│   ├── detection/           # YOLOv8 detector
│   │   └── yolo_detector.py
│   ├── tracking/            # DeepSORT tracker
│   │   └── deep_sort_tracker.py
│   ├── analytics/          # Analytics engine
│   │   └── analytics_engine.py
│   ├── anomaly/            # Anomaly detection
│   │   └── anomaly_detector.py
│   ├── gesture/            # Gesture recognition
│   │   └── gesture_recognizer.py
│   ├── ui/                 # Dashboard and visualization
│   │   ├── dashboard.py
│   │   └── visualizer.py
│   └── utils/              # Utility functions
│       └── config_loader.py
└── README.md
```

## Troubleshooting

### CUDA/GPU Issues
- If CUDA is not available, the system will automatically use CPU (slower)
- Ensure PyTorch is installed with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Video Input Issues
- Check webcam index in config (default is 0)
- For video files, ensure path is correct and file format is supported
- Common formats: .mp4, .avi, .mov

### Performance Optimization
- Use smaller YOLOv8 model (yolov8n, yolov8s) for faster processing
- Reduce frame resolution in config
- Disable unnecessary analytics features

## License

This project is provided as-is for educational and research purposes.
