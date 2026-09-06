"""
🚀 YOLOv8 Drowsiness Detection System - Web Application
Streamlit Cloud & Docker Ready Web Interface
"""

import os
import time
import base64
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from detector import DrowsinessDetector

# Set page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Futuristic Driver HUD Dark Theme
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
        color: #e6edf3;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 50%, #388bfd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
    }
    
    /* Status Badges */
    .status-badge {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 15px;
        animation: pulse 2s infinite;
    }
    
    .status-normal {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
    }
    
    .status-warning {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #e3b341;
    }
    
    .status-danger {
        background: rgba(248, 81, 73, 0.2);
        border: 1px solid #f85149;
        color: #ff7b72;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


# Load audio buzzer for web playback
@st.cache_data
def get_buzzer_audio_base64() -> str:
    buzzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buzzer.mp3")
    if os.path.exists(buzzer_path):
        try:
            with open(buzzer_path, "rb") as f:
                data = f.read()
                return base64.b64encode(data).decode()
        except Exception:
            pass
    return ""


BUZZER_B64 = get_buzzer_audio_base64()


def trigger_web_buzzer():
    """Play buzzer alarm in client browser."""
    if BUZZER_B64:
        audio_html = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{BUZZER_B64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        # Web Audio API Synthesizer fallback
        beep_html = """
        <script>
            (function() {
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(800, ctx.currentTime);
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.3);
                } catch(e) {}
            })();
        </script>
        """
        st.markdown(beep_html, unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/car-service.png", width=64)
    st.title("⚙️ Detection Settings")
    
    st.markdown("---")
    st.subheader("Sensitivity Controls")
    
    eye_thresh = st.slider(
        "Eye Closure Threshold (Frames)",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Consecutive closed-eye frames needed to trigger Drowsiness Alert."
    )
    
    yawn_thresh = st.slider(
        "Yawn Threshold (Frames)",
        min_value=5,
        max_value=120,
        value=35,
        step=5,
        help="Consecutive yawn frames needed to trigger Fatigue Warning."
    )
    
    conf_eye = st.slider(
        "Eye Detection Confidence",
        min_value=0.15,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Lower confidence improves sensitivity in poor cabin lighting."
    )
    
    conf_yawn = st.slider(
        "Yawn Detection Confidence",
        min_value=0.20,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Yawn model detection threshold."
    )
    
    st.markdown("---")
    st.subheader("Alarm & Audio")
    enable_audio = st.toggle("Enable Audible Buzzer Alarm", value=True)
    
    st.markdown("---")
    st.subheader("🧠 Active AI Models")
    
    # Initialize or refresh detector in session state
    if "detector" not in st.session_state or st.session_state.detector.using_fallback:
        st.session_state.detector = DrowsinessDetector(
            eye_closure_threshold=eye_thresh,
            yawn_threshold=yawn_thresh
        )
    else:
        st.session_state.detector.eye_closure_threshold = eye_thresh
        st.session_state.detector.yawn_threshold = yawn_thresh
        
    eye_ok = st.session_state.detector.eye_model is not None
    yawn_ok = st.session_state.detector.yawn_model is not None
    
    st.markdown(f"**Eye Model:** {'🟢 Active (YOLOv8x - 136MB)' if eye_ok else '🔴 Not Found'}")
    st.markdown(f"**Yawn Model:** {'🟢 Active (YOLOv8x - 136MB)' if yawn_ok else '🔴 Not Found'}")
    
    if st.button("🔄 Reload Model Weights", use_container_width=True):
        st.session_state.detector = DrowsinessDetector(
            eye_closure_threshold=eye_thresh,
            yawn_threshold=yawn_thresh
        )
        st.success("Models reloaded successfully!")
        st.rerun()

    st.markdown("---")
    st.caption("🚀 Powered by Ultralytics YOLOv8 & PyTorch")


# Hero Header
st.markdown("""
<div class="hero-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 class="hero-title">🚗 AI Driver Drowsiness & Fatigue Detection</h1>
            <p class="hero-subtitle">Real-time computer vision monitoring with YOLOv8 eye tracking, yawn analysis, and instant audible alert system.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Top-level Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Live Camera Monitor",
    "🎬 Video File Analysis",
    "🖼️ Image Snapshot Test",
    "📊 System Telemetry & Architecture"
])


# ==========================================
# TAB 1: LIVE CAMERA MONITOR
# ==========================================
with tab1:
    st.subheader("Live Driver Monitoring Feed")
    
    cam_mode = st.radio(
        "Select Camera Input Mode:",
        ["🔴 Continuous Live Webcam Stream (Real-Time)", "📸 Camera Snapshot Mode"],
        horizontal=True
    )
    
    col_cam, col_hud = st.columns([3, 2])
    
    metrics = st.session_state.detector.get_metrics()
    
    with col_cam:
        if cam_mode == "🔴 Continuous Live Webcam Stream (Real-Time)":
            st.write("Streams your physical camera continuously frame-by-frame with zero delay.")
            run_live = st.checkbox("▶️ Start Live Continuous Monitoring", value=False)
            video_feed = st.empty()
            
            if run_live:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("⚠️ Could not open local webcam (device index 0). If you are accessing remotely or in cloud, use 'Camera Snapshot Mode'.")
                else:
                    st.info("💡 Real-time monitoring active. Uncheck 'Start Live Continuous Monitoring' to stop.")
                    while run_live:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab frame from camera.")
                            break
                            
                        # Process frame
                        ann_frame, metrics = st.session_state.detector.process_frame(
                            frame,
                            conf_eye=conf_eye,
                            conf_yawn=conf_yawn,
                            draw_overlay=True
                        )
                        
                        # Convert to RGB and display
                        frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                        video_feed.image(frame_rgb, use_container_width=True)
                        
                        # Alarm check
                        if metrics["alert_status"] == "DANGER" and enable_audio:
                            trigger_web_buzzer()
                            
                    cap.release()
            else:
                st.info("Check the box above to start live continuous webcam stream.")
                
        else:
            cam_image = st.camera_input("Capture a driver camera snapshot:")
            if cam_image is not None:
                file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                ann_frame, metrics = st.session_state.detector.process_frame(
                    frame,
                    conf_eye=conf_eye,
                    conf_yawn=conf_yawn,
                    draw_overlay=True
                )
                
                frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Processed Driver Snapshot", use_container_width=True)
                
                if metrics["alert_status"] == "DANGER" and enable_audio:
                    trigger_web_buzzer()
            else:
                st.info("💡 Click the button above to capture a camera snapshot.")
                
    with col_hud:
        st.markdown("### 🚦 Driver Status HUD")
        
        # Status Card
        status = metrics.get("alert_status", "NORMAL")
        if status == "DANGER":
            st.markdown("""
            <div class="status-badge status-danger">
                🚨 CRITICAL: DROWSINESS DETECTED!<br><span style="font-size:0.85rem">PULL OVER AND TAKE A BREAK</span>
            </div>
            """, unsafe_allow_html=True)
        elif status == "WARNING":
            st.markdown("""
            <div class="status-badge status-warning">
                ⚠️ WARNING: FATIGUE DETECTED<br><span style="font-size:0.85rem">Driver showing early signs of sleepiness</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-normal">
                ✅ STATUS: DRIVER ALERT & ACTIVE
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Grid
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('eye_closure_count', 0)}</div>
                <div class="metric-label">Eye Closure Counter</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('yawn_count', 0)}</div>
                <div class="metric-label">Yawn Counter</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'CLOSED' if metrics.get('eyes_closed') else 'OPEN'}</div>
                <div class="metric-label">Current Eye State</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'YES' if metrics.get('is_yawning') else 'NO'}</div>
                <div class="metric-label">Current Yawn State</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset HUD Counters", use_container_width=True):
            st.session_state.detector.reset_state()
            st.rerun()


# ==========================================
# TAB 2: VIDEO FILE ANALYSIS
# ==========================================
with tab2:
    st.subheader("Upload & Analyze Driving Video Footage")
    st.write("Upload an MP4, AVI, or MOV video (e.g. dashcam or driver cabin recording) to run deep learning analysis.")
    
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        st.write(f"📊 **Video Info:** {total_frames} total frames | {fps:.1f} FPS")
        
        if st.button("🚀 Start Video Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Temporary output video path
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_HEIGHT)) or 480
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(out_path, fourcc, min(fps, 30), (width, height))
            
            local_detector = DrowsinessDetector(
                eye_closure_threshold=eye_thresh,
                yawn_threshold=yawn_thresh
            )
            
            timeline_data = []
            curr_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                curr_frame += 1
                
                # Run inference
                ann_frame, m = local_detector.process_frame(
                    frame,
                    conf_eye=conf_eye,
                    conf_yawn=conf_yawn,
                    draw_overlay=True
                )
                
                out_writer.write(ann_frame)
                
                timeline_data.append({
                    "Frame": curr_frame,
                    "Timestamp (s)": round(curr_frame / fps, 2),
                    "Eye Closure Count": m["eye_closure_count"],
                    "Yawn Count": m["yawn_count"],
                    "Alert Status": m["alert_status"]
                })
                
                if curr_frame % 5 == 0 or curr_frame == total_frames:
                    progress = min(1.0, curr_frame / max(1, total_frames))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {curr_frame}/{total_frames} ({int(progress*100)}%)")
                    
            cap.release()
            out_writer.release()
            
            status_text.success("✅ Video Analysis Complete!")
            
            # Render analytics chart
            df = pd.DataFrame(timeline_data)
            if not df.empty:
                st.subheader("📈 Fatigue Timeline Telemetry")
                fig = px.line(
                    df,
                    x="Timestamp (s)",
                    y=["Eye Closure Count", "Yawn Count"],
                    title="Driver Fatigue Metrics Over Time",
                    color_discrete_map={"Eye Closure Count": "#58a6ff", "Yawn Count": "#f85149"},
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download CSV report
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Fatigue Telemetry CSV",
                    data=csv,
                    file_name="driver_fatigue_telemetry.csv",
                    mime="text/csv"
                )


# ==========================================
# TAB 3: IMAGE SNAPSHOT TEST
# ==========================================
with tab3:
    st.subheader("Single Image Fatigue Test")
    st.write("Upload an image of a driver to check eye state and yawn detection.")
    
    uploaded_img = st.file_uploader("Upload Driver Photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        img_pil = Image.open(uploaded_img)
        img_np = np.array(img_pil)
        
        # Convert RGB to BGR for detector
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np
            
        test_detector = DrowsinessDetector(
            eye_closure_threshold=eye_thresh,
            yawn_threshold=yawn_thresh
        )
        
        annotated_bgr, metrics = test_detector.process_frame(
            img_bgr,
            conf_eye=conf_eye,
            conf_yawn=conf_yawn,
            draw_overlay=True
        )
        
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        col_orig, col_res = st.columns(2)
        with col_orig:
            st.image(img_pil, caption="Original Input Image", use_container_width=True)
        with col_res:
            st.image(annotated_rgb, caption="YOLOv8 Detection Result", use_container_width=True)
            
        st.subheader("🔍 Detection Breakdown")
        st.json(metrics)


# ==========================================
# TAB 4: ARCHITECTURE & TELEMETRY
# ==========================================
with tab4:
    st.subheader("System Architecture & Deep Learning Models")
    
    st.markdown("""
    ### 🧠 How the Drowsiness Detection System Works
    
    1. **Dual YOLOv8 Models**:
       - **Eye State Model**: Detects `Closed` vs `Open` eyes with high precision and low latency.
       - **Yawn Detection Model**: Identifies mouth yawning postures indicative of microsleeps and oxygen deprivation.
    2. **Stateful Alert Logic**:
       - Multi-frame rolling counter tracks prolonged eyelid closure.
       - Evaluates cumulative yawning events over time to compute driver fatigue severity.
       - Triggers three progressive states: **NORMAL** ➔ **WARNING** ➔ **CRITICAL DANGER**.
    3. **Cross-Platform Audible Buzzer**:
       - Built-in HTML5 Web Audio and MP3 buzzer synthesizer triggers alarms directly in the client device's speakers.
    4. **Cloud & Edge Readiness**:
       - Headless OpenCV compatible.
       - 1-Click deployable on Streamlit Cloud, Hugging Face Spaces, Render, and Docker.
    """)
    
    st.markdown("---")
    st.caption("System Version: 2.1.0 (Production Cloud Build) | YOLOv8 Powered")
