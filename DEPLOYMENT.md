# 🚀 Deployment Guide: YOLOv8 Drowsiness Detection System

This guide outlines multiple deployment pathways for hosting your **Driver Drowsiness and Fatigue Detection System**, ranging from **free 1-click cloud platforms** to **Docker containers**.

---

## 📋 Table of Contents
1. [Option 1: Streamlit Community Cloud (Recommended - 100% Free)](#1-streamlit-community-cloud-free--instant)
2. [Option 2: Hugging Face Spaces (Free Cloud AI Hosting)](#2-hugging-face-spaces-free)
3. [Option 3: Render (Free Web Service)](#3-render-free-web-service)
4. [Option 4: Docker & Docker Compose (Self-Hosted / Cloud VM)](#4-docker--docker-compose)
5. [Option 5: Local Desktop & Network Run](#5-local-desktop--network-run)

---

## 1. Streamlit Community Cloud (Free & Instant)

Streamlit Community Cloud allows you to deploy the interactive web app directly from your GitHub repository for free with zero server setup.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add cloud deployment configuration"
   git push origin main
   ```
2. **Open Streamlit Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. **Create New App**:
   - Click **"New App"**.
   - Select your repository: `Drousiness-Detection-System-Using-YOLOv8`.
   - Branch: `main`.
   - Main file path: `app.py`.
4. **Deploy**:
   - Click **"Deploy"**. Your live web app with real-time camera support will be ready at a public URL in 2–3 minutes!

---

## 2. Hugging Face Spaces (Free)

Hugging Face Spaces provides high-performance cloud hosting for AI models.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set Space SDK to **"Streamlit"**.
3. Choose **Public** or **Private** and select the free CPU hardware tier.
4. Clone your Space repository locally or push your files:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

---

## 3. Render (Free Web Service)

Render allows you to host web services and Docker containers.

### Steps:
1. Push this repository to GitHub.
2. Go to [render.com](https://render.com/) and click **"New +" ➔ "Web Service"**.
3. Connect your GitHub repository.
4. Fill in the deployment details:
   - **Environment**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Click **"Create Web Service"**.

---

## 4. Docker & Docker Compose

Docker packages all PyTorch, Ultralytics, OpenCV, and audio dependencies into a portable container.

### Build and Run with Docker:
```bash
# Build the Docker image
docker build -t drowsiness-detector:latest .

# Run the Streamlit Web Application
docker run -d -p 8501:8501 --name drowsiness-web drowsiness-detector:latest

# Open in browser: http://localhost:8501
```

### Run with Docker Compose:
```bash
docker compose up -d
```
- **Streamlit Web Dashboard**: `http://localhost:8501`

---

## 5. Local Desktop & Network Run

### Run the Web Dashboard:
```bash
streamlit run app.py
```
To allow other devices (smartphones, tablets in the car) on your local Wi-Fi to connect, use the **Network URL** shown in your terminal (e.g. `http://192.168.1.X:8501`).

### Run the Standalone OpenCV Window:
```bash
python main.py
```

