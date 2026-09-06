# 🚗 Driver Drowsiness & Fatigue Detection System (YOLOv8) 💤

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, real-time computer vision system built with **Ultralytics YOLOv8** designed to monitor driver alertness, track eye closures and yawning frequencies, and trigger audible alarms to prevent fatigue-related road accidents.

---

## 🌟 Key Features

- 👁️ **Eye State Detection**: High-accuracy classification of `Open` vs `Closed` eyes across diverse lighting conditions.
- 🥱 **Yawn & Fatigue Tracking**: Detects yawning patterns and calculates cumulative driver exhaustion levels.
- 🚦 **Real-Time Driver HUD**: Live heads-up display showing stateful alert badges (`NORMAL`, `WARNING`, `CRITICAL DANGER`).
- 🔊 **Cross-Platform Audible Buzzer**: Synthesizes and plays audible warning alarms in any web browser or native client.
- 📱 **Interactive Web Dashboard**: Streamlit-powered dark UI for live browser camera feeds, video file analysis, and image tests.
- 🐳 **Cloud & Docker Ready**: 1-Click deployable on Streamlit Cloud, Hugging Face Spaces, Render, and Docker.

---

## 📁 Repository Structure

```text
├── app.py                   # 🌐 Interactive Streamlit Web Application & HUD
├── detector.py              # 🧠 Core YOLOv8 Drowsiness & Fatigue Detection Engine
├── main.py                  # 💻 Local Desktop OpenCV Webcam loop
├── buzzer.mp3               # 🔊 Audio alarm sound asset
├── requirements.txt         # 📦 Production Python dependencies
├── Dockerfile               # 🐳 Optimized container build definition
├── docker-compose.yml       # 🚢 Multi-service container orchestration
├── render.yaml              # ☁️ 1-Click Render Cloud deployment config
├── DEPLOYMENT.md            # 🚀 Step-by-step multi-platform deployment guide
├── datasets/                # 🗃️ Labeled dataset (Eye & Yawn annotations)
├── model/                   # 🏋️ YOLOv8 weights and training configurations
└── README.md                # 📖 Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/rishavraj69/Drousiness-Detection-System-Using-YOLOv8.git
cd Drousiness-Detection-System-Using-YOLOv8
pip install -r requirements.txt
```

### 2. Run the Interactive Web App (Recommended)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to access the live camera monitor, upload driving videos, and view real-time fatigue telemetry.

### 3. Run with Docker
```bash
# Start Web Dashboard
docker compose up -d
```

---

## ☁️ Deployment Guides

For complete, step-by-step instructions on deploying this system to **Streamlit Community Cloud (Free)**, **Hugging Face Spaces**, **Render**, or **Cloud VMs (AWS / GCP / Azure)**, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📊 System Architecture & Metrics

| State | Condition | Trigger Action |
| :--- | :--- | :--- |
| 🟢 **NORMAL** | Eyes open, normal blink cycle | Safe status badge displayed |
| 🟡 **WARNING** | Frequent yawning or brief eye closures | Warning badge, advisory message |
| 🔴 **CRITICAL DANGER** | Eye closure > 40 frames or cumulative fatigue | Loud audible alarm, emergency alert |

---

## 📜 License
This project is open-source under the MIT License.
