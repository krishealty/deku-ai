# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: For GPU acceleration, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Configure System

Edit `config.yaml` to set your input source:

### For Webcam:
```yaml
input:
  source: "webcam"
  webcam_index: 0  # Change if using different camera
```

### For Video File:
```yaml
input:
  source: "video_file"
  video_path: "path/to/your/video.mp4"
```

## Step 3: Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Step 4: Using the System

### Dashboard Controls
- Use the sidebar to switch between display modes
- Toggle analytics features on/off
- Adjust detection thresholds
- Enable/disable anomaly detection types

### Gesture Control
1. Enable "Gesture Control" in the sidebar
2. Hold up fingers in front of the camera:
   - 1 finger: Vehicle density view
   - 2 fingers: Pedestrian density view
   - 3 fingers: Heatmap overlay
   - 4 fingers: Incident detection mode
   - 5 fingers: Full analytics view

### Understanding the Display
- **Bounding Boxes**: Colored boxes around detected objects
  - Cyan: Vehicles
  - Magenta: Pedestrians
- **Track IDs**: Numbers above boxes show persistent object IDs
- **Trajectories**: Paths showing object movement (if enabled)
- **Alerts**: Red/orange markers indicate anomalies
- **Metrics**: Real-time counts and statistics in top-right

## Troubleshooting

### "Failed to initialize video"
- Check webcam is connected and not in use by another application
- Verify video file path is correct
- Try changing `webcam_index` in config

### Low FPS / Slow Performance
- Use smaller YOLOv8 model (change `model.type` to `yolov8s` or `yolov8n`)
- Reduce frame resolution in config
- Ensure GPU is being used (check system status in sidebar)

### No Detections
- Lower confidence threshold in sidebar
- Ensure good lighting and clear view
- Check that objects match COCO classes (vehicles, pedestrians)

### CUDA Errors
- System will auto-fallback to CPU
- For GPU: Verify CUDA installation and PyTorch CUDA support

## Next Steps

- Customize flow lines in `config.yaml` for your scene
- Adjust anomaly detection thresholds for your use case
- Calibrate `pixels_per_meter` for accurate speed estimation
- Add more flow lines for complex intersections

