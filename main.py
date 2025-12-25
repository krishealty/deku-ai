"""Main application for Traffic Intelligence System."""
import cv2
import numpy as np
import threading
import time
from queue import Queue
from typing import Optional, Dict
import streamlit as st

from src.utils.config_loader import load_config, get_device
from src.detection.yolo_detector import YOLODetector
from src.tracking.deep_sort_tracker import DeepSORTTracker
from src.analytics.analytics_engine import AnalyticsEngine
from src.anomaly.anomaly_detector import AnomalyDetector
from src.gesture.gesture_recognizer import GestureRecognizer
from src.ui.visualizer import Visualizer
from src.ui.dashboard import Dashboard


class TrafficIntelligenceSystem:
    """Main system class integrating all components."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the traffic intelligence system."""
        # Load configuration
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        
        # Initialize components
        model_config = self.config.get('model', {})
        self.detector = YOLODetector(
            model_type=model_config.get('type', 'yolov8l'),
            device=self.device,
            confidence_threshold=model_config.get('confidence_threshold', 0.5),
            iou_threshold=model_config.get('iou_threshold', 0.45)
        )
        
        tracking_config = self.config.get('tracking', {})
        self.tracker = DeepSORTTracker(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3)
        )
        
        # Get frame dimensions from config
        input_config = self.config.get('input', {})
        self.frame_width = input_config.get('frame_width', 1280)
        self.frame_height = input_config.get('frame_height', 720)
        
        analytics_config = self.config.get('analytics', {})
        self.analytics = AnalyticsEngine(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            grid_size=analytics_config.get('density_grid_size', 10),
            pixels_per_meter=analytics_config.get('speed_estimation', {}).get('pixels_per_meter', 10.0),
            fps=analytics_config.get('speed_estimation', {}).get('fps', 30.0)
        )
        
        # Setup flow lines from config
        flow_lines_config = analytics_config.get('flow_lines', [])
        for flow_line_config in flow_lines_config:
            self.analytics.add_flow_line(
                name=flow_line_config.get('name', 'line_1'),
                start=tuple(flow_line_config.get('start', [0.2, 0.5])),
                end=tuple(flow_line_config.get('end', [0.8, 0.5])),
                direction=flow_line_config.get('direction', 'horizontal')
            )
        
        self.anomaly_detector = AnomalyDetector(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=analytics_config.get('speed_estimation', {}).get('fps', 30.0)
        )
        
        gesture_config = self.config.get('gesture_control', {})
        if gesture_config.get('enabled', True):
            self.gesture_recognizer = GestureRecognizer(
                min_detection_confidence=gesture_config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=gesture_config.get('min_tracking_confidence', 0.5),
                gesture_hold_time=gesture_config.get('gesture_hold_time', 1.0)
            )
        else:
            self.gesture_recognizer = None
        
        self.visualizer = Visualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        # Video input
        self.input_source = input_config.get('source', 'webcam')
        self.video_cap = None
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.current_frame = None
        self.current_tracks = []
        self.current_metrics = {}
        self.current_alerts = []
        self.current_heatmap = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
    
    def initialize_video(self):
        """Initialize video input source."""
        input_config = self.config.get('input', {})
        
        if self.input_source == 'webcam':
            webcam_index = input_config.get('webcam_index', 0)
            self.video_cap = cv2.VideoCapture(webcam_index)
            if self.frame_width and self.frame_height:
                self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        elif self.input_source == 'video_file':
            video_path = input_config.get('video_path', '')
            if video_path:
                self.video_cap = cv2.VideoCapture(video_path)
            else:
                raise ValueError("Video path not specified in config")
        else:
            raise ValueError(f"Unknown input source: {self.input_source}")
        
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.input_source}")
        
        # Update actual frame dimensions
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline."""
        current_time = time.time()
        
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracks = self.tracker.update(detections)
        self.current_tracks = tracks
        
        # Analytics
        metrics = self.analytics.get_all_metrics(tracks, current_time)
        self.current_metrics = metrics
        
        # Anomaly detection
        previous_density = self.analytics.metrics_history.get('pedestrian_density', [0])
        prev_dens = previous_density[-1] if previous_density else 0.0
        alerts = self.anomaly_detector.detect_all(tracks, current_time, prev_dens)
        self.current_alerts = [alert.__dict__ for alert in alerts]
        
        # Get heatmap
        self.current_heatmap = self.analytics.get_heatmap()
        
        # Gesture recognition (if enabled)
        gesture_info = {}
        if self.gesture_recognizer:
            gesture_info = self.gesture_recognizer.process_frame(frame)
        
        # Visualization
        output_frame = frame.copy()
        
        # Apply visualizations based on display mode
        # (This will be controlled by dashboard UI)
        output_frame = self.visualizer.draw_bounding_boxes(
            output_frame, tracks, show_ids=True, show_category=True
        )
        
        if self.current_heatmap is not None:
            output_frame = self.visualizer.draw_heatmap(output_frame, self.current_heatmap, alpha=0.3)
        
        # Draw flow lines
        if hasattr(self.analytics, 'flow_lines'):
            output_frame = self.visualizer.draw_flow_lines(output_frame, self.analytics.flow_lines)
        
        # Draw alerts
        if self.current_alerts:
            output_frame = self.visualizer.draw_alerts(output_frame, self.current_alerts)
        
        # Draw metrics overlay
        output_frame = self.visualizer.draw_metrics_overlay(output_frame, metrics)
        
        # Draw gesture info
        if self.gesture_recognizer and gesture_info.get('hand_detected'):
            output_frame = self.gesture_recognizer.draw_gesture(output_frame, gesture_info)
        
        # Update FPS
        self.frame_count += 1
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        return output_frame
    
    def run_streamlit_app(self):
        """Run the Streamlit dashboard application."""
        dashboard = Dashboard()
        
        # Initialize session state
        if 'system_initialized' not in st.session_state:
            try:
                self.initialize_video()
                st.session_state.system_initialized = True
                st.session_state.video_cap = self.video_cap
            except Exception as e:
                st.error(f"Failed to initialize video: {e}")
                return
        else:
            self.video_cap = st.session_state.video_cap
        
        # Main app
        st.title("🚦 Urban Traffic Intelligence System")
        
        # Sidebar controls
        controls = dashboard.render_sidebar(self.config)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Video Feed")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("Metrics")
            metrics_placeholder = st.empty()
            
            st.subheader("Alerts")
            alerts_placeholder = st.empty()
        
        # Process single frame
        ret, frame = self.video_cap.read()
        if ret:
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update video display
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Update metrics
            with metrics_placeholder.container():
                dashboard.render_metrics_cards(self.current_metrics)
            
            # Update alerts
            with alerts_placeholder.container():
                dashboard.render_alerts_panel(self.current_alerts)
            
            # Auto-refresh for live video
            time.sleep(0.033)  # ~30 FPS
            st.rerun()
        else:
            st.warning("Failed to read frame. Check video source.")
    
    def run_opencv_display(self):
        """Run with OpenCV display (alternative to Streamlit)."""
        self.initialize_video()
        self.running = True
        
        print("Starting traffic intelligence system...")
        print("Press 'q' to quit")
        
        try:
            while self.running:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display FPS
                cv2.putText(
                    processed_frame,
                    f"FPS: {self.current_fps:.1f}",
                    (10, processed_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Show frame
                cv2.imshow("Traffic Intelligence System", processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
        
        finally:
            self.video_cap.release()
            cv2.destroyAllWindows()
            print("System stopped.")


def main():
    """Main entry point."""
    import sys
    
    # Check if running in Streamlit
    if 'streamlit' in sys.modules or hasattr(sys, '_getframe'):
        try:
            # Try to detect Streamlit context
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                # Running in Streamlit
                system = TrafficIntelligenceSystem()
                system.run_streamlit_app()
                return
        except:
            pass
    
    # Default: Run with Streamlit
    # For command-line usage, uncomment the OpenCV version:
    # system = TrafficIntelligenceSystem()
    # system.run_opencv_display()
    
    # Streamlit entry point
    system = TrafficIntelligenceSystem()
    system.run_streamlit_app()


if __name__ == "__main__":
    main()

