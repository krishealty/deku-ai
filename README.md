# 🚦 DEKU  
## Deep Environmental Knowledge for Urban Traffic

DEKU is a **vision-driven smart city intelligence platform** that transforms live CCTV or webcam feeds into **real-time traffic, pedestrian, and incident analytics**. It leverages advanced computer vision, multi-object tracking, and AI-powered analysis to help cities **understand, monitor, and respond to urban movement** more effectively.

---

## 🌆 Problem Statement

- Traffic cameras generate **raw video without actionable insights**
- Manual monitoring is **slow, expensive, and error-prone**
- Most systems focus only on **vehicles**, ignoring pedestrian flow
- Traffic congestion and incidents are detected **too late**
- Existing dashboards require **constant manual interaction**

**Cities see traffic, but they don’t understand it in real time.**

---

## 🚀 Solution Overview

DEKU converts live video streams into **urban intelligence** by detecting, tracking, and analyzing both vehicles and pedestrians. It provides real-time density estimation, flow analysis, heatmaps, and incident detection through a **cinematic dashboard**, with support for both **traditional UI controls** and **gesture-based interaction**.

---

## ✨ Key Features

### 🚗 Traffic & Crowd Analytics
- Real-time vehicle detection (cars, buses, trucks, motorcycles, bicycles)
- Real-time pedestrian and crowd detection
- Multi-object tracking with persistent IDs
- Vehicle and pedestrian density estimation
- Traffic flow rate and speed analysis
- Crowd build-up and loitering detection

### 🚨 Incident & Anomaly Detection
- Sudden vehicle stoppage detection
- Potential collision and congestion detection
- Wrong-way movement detection
- Abnormal pedestrian crowd surges

### 🖐️ Interaction & Control
- Gesture-based control using computer vision
- Finger-count gestures to switch analytics modes
- Traditional dashboard controls (buttons, toggles, sliders)
- Seamless switching between control methods

### 🖥️ Visualization & Dashboard
- Live video feed with AI overlays
- Real-time metrics and counters
- Traffic heatmaps and time-series charts
- Incident alerts and system status panels
- Dark, futuristic, smart-city-inspired UI

---

## 🧠 Technology Stack

### Computer Vision & AI
- YOLOv8 (vehicle and pedestrian detection)
- Multi-object tracking (DeepSORT / ByteTrack)
- MediaPipe (gesture recognition)

### Backend
- Python
- FastAPI (REST & WebSocket APIs)

### Frontend
- React (or similar modern framework)
- Tailwind CSS / modern UI components
- Chart.js / Recharts for analytics

### Performance
- GPU-accelerated inference (CUDA-supported)
- Optimized for real-time processing

---

## 🌍 Google AI & Cloud Integration

- **MediaPipe** – Gesture-based interaction
- **Google Cloud Vision AI** – Detection validation (optional)
- **Vertex AI** – Model training and experimentation
- **Firebase** – Real-time data synchronization
- **Google Maps Platform** – Geographic traffic visualization

---

## 🏗️ System Architecture
