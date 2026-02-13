import cv2
import time
import threading
import queue
import os
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
RTSP_URL = "rtsp://10.76.11.62"

# --- Global Resources ---
print("[System] Loading YOLO Model globally...")
try:
    model = YOLO("best_m.pt")
except:
    print("[Warning] 'best_m.pt' not found, falling back to 'yolov8n.pt'")
    model = YOLO("yolov8n.pt")

model_lock = threading.Lock()

# --- Shared State ---
class VideoState:
    def __init__(self):
        self.frame = None
        self.boxes = []  
        self.count = 0
        self.lock = threading.Lock()
        self.running = False 

state = VideoState()

# --- Thread Functions ---
def capture_loop():
    print(f"[Thread 1] Connecting to Stream: {RTSP_URL}...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("Stream interrupted. Reconnecting in 2s...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        with state.lock:
            state.frame = frame
    
    cap.release()
    print("[Thread 1] Stopped.")

def ai_loop():
    print("[Thread 2] AI Processing Started...")
    ai_width = 640
    ai_height = 360

    while state.running:
        working_frame = None
        with state.lock:
            if state.frame is not None:
                working_frame = state.frame.copy()

        if working_frame is None:
            time.sleep(0.05)
            continue

        orig_h, orig_w = working_frame.shape[:2]
        input_frame = cv2.resize(working_frame, (ai_width, ai_height))
        
        with model_lock:
            results = model(input_frame, conf=0.5, classes=[0], verbose=False)
        
        current_boxes = []
        if len(results) > 0:
            det_boxes = results[0].boxes.xyxy.cpu().numpy()
            x_scale = orig_w / ai_width
            y_scale = orig_h / ai_height

            for box in det_boxes:
                x1, y1, x2, y2 = box
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)
                current_boxes.append((x1, y1, x2, y2))

        with state.lock:
            state.count = len(current_boxes)
            state.boxes = current_boxes
            
        time.sleep(0.01) 
    print("[Thread 2] Stopped.")

# --- Flask Routes ---

@app.route('/start_stream', methods=['POST'])
def start_stream():
    if not state.running:
        state.running = True
        threading.Thread(target=capture_loop, daemon=True).start()
        threading.Thread(target=ai_loop, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    if state.running:
        state.running = False 
        time.sleep(0.5)
        with state.lock:
            state.frame = None
            state.count = 0
            state.boxes = []
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with model_lock:
            results = model(img, conf=0.5, classes=[0], verbose=False)
        count = len(results[0].boxes)
        return jsonify({"message": "Success", "detected_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate_annotated_feed():
        while True:
            if not state.running:
                break 
            
            output_frame = None
            current_boxes = []

            with state.lock:
                if state.frame is None:
                    time.sleep(0.1)
                    continue
                output_frame = state.frame.copy()
                current_boxes = list(state.boxes)

            # Draw Cyberpunk Style Boxes
            for (x1, y1, x2, y2) in current_boxes:
                # Corners only for cleaner look? Or full box. Let's do full box thin.
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # Label
                label = "TARGET"
                cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', output_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
            
    return Response(generate_annotated_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_feed')
def count_feed():
    def generate_sse_count():
        last_count = -1
        while True:
            if not state.running:
                yield f"data: 0\n\n"
                break
            with state.lock:
                c = state.count
            if c != last_count:
                yield f"data: {c}\n\n"
                last_count = c
            time.sleep(0.5)
    return Response(generate_sse_count(), mimetype='text/event-stream')

# --- Frontend Template ---
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CrowdSentinel — Live Risk Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet"/>
  
  <style>
    :root {
        --c-bg: #050505;
        --c-panel: #0a0a0a;
        --c-accent: #00ffff;
        --c-accent-dim: rgba(0, 255, 255, 0.1);
        --c-danger: #ff003c;
        --c-text: #e0e0e0;
        --font-mono: 'Share Tech Mono', monospace;
        --font-ui: 'Barlow Condensed', sans-serif;
    }

    * { box-sizing: border-box; }
    body {
        margin: 0;
        background-color: var(--c-bg);
        color: var(--c-text);
        font-family: var(--font-ui);
        overflow-x: hidden;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    /* Scanlines */
    .scanlines {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.2) 50%, rgba(0,0,0,0.2));
        background-size: 100% 4px;
        pointer-events: none;
        z-index: 10;
        opacity: 0.6;
    }

    /* Header */
    .header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 30px;
        background: var(--c-panel);
        border-bottom: 1px solid #333;
    }
    .header-left { display: flex; align-items: center; gap: 15px; }
    .logo-mark { display: flex; gap: 3px; }
    .logo-dot { width: 6px; height: 6px; background: var(--c-accent); border-radius: 50%; }
    .logo-text { font-family: var(--font-mono); font-size: 24px; margin: 0; letter-spacing: 2px; }
    .logo-text span { color: var(--c-accent); }
    .logo-sub { font-size: 10px; color: #666; letter-spacing: 1px; margin: 0; text-transform: uppercase; }
    
    .live-badge { 
        display: flex; align-items: center; gap: 8px; 
        font-family: var(--font-mono); color: var(--c-danger);
        border: 1px solid var(--c-danger); padding: 5px 10px; border-radius: 4px;
        text-shadow: 0 0 5px var(--c-danger);
    }
    .pulse-ring { width: 8px; height: 8px; background: var(--c-danger); border-radius: 50%; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

    /* Layout */
    .main-grid {
        display: grid;
        grid-template-columns: 1fr 300px;
        grid-template-rows: auto 1fr;
        gap: 20px;
        padding: 20px;
        flex: 1;
        height: calc(100vh - 80px);
    }

    /* Panels */
    .panel { background: var(--c-panel); border: 1px solid #333; position: relative; display: flex; flex-direction: column; }
    .panel-label { 
        background: #111; padding: 8px 15px; font-family: var(--font-mono); font-size: 12px; 
        color: #888; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 8px;
    }
    .label-icon { color: var(--c-accent); }

    /* Video Feed */
    .panel--feed { grid-column: 1; grid-row: 1 / span 2; overflow: hidden; }
    .feed-wrapper { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; background: #000; }
    #videoFeed { max-width: 100%; max-height: 100%; width: 100%; object-fit: contain; opacity: 0.2; transition: opacity 0.5s; }
    #videoFeed.active { opacity: 1; }
    
    .feed-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; padding: 20px; box-sizing: border-box; }
    .corner { position: absolute; width: 20px; height: 20px; border: 2px solid var(--c-accent); transition: all 0.3s; }
    .tl { top: 20px; left: 20px; border-right: none; border-bottom: none; }
    .tr { top: 20px; right: 20px; border-left: none; border-bottom: none; }
    .bl { bottom: 20px; left: 20px; border-right: none; border-top: none; }
    .br { bottom: 20px; right: 20px; border-left: none; border-top: none; }

    /* Sidebar */
    .sidebar { display: flex; flex-direction: column; gap: 15px; overflow-y: auto; }
    .card { background: #111; border: 1px solid #333; padding: 15px; }
    .card__label { font-family: var(--font-mono); color: #666; font-size: 12px; margin-bottom: 10px; }
    
    .count-display { display: flex; align-items: baseline; gap: 5px; }
    .count-number { font-size: 48px; font-weight: 700; color: #fff; line-height: 1; }
    .count-unit { font-size: 14px; color: #888; font-family: var(--font-mono); }
    
    .btn-cyber {
        width: 100%;
        background: var(--c-accent-dim);
        border: 1px solid var(--c-accent);
        color: var(--c-accent);
        padding: 12px;
        font-family: var(--font-mono);
        cursor: pointer;
        text-transform: uppercase;
        transition: all 0.3s;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .btn-cyber:hover { background: var(--c-accent); color: #000; box-shadow: 0 0 15px var(--c-accent); }
    .btn-cyber.danger { border-color: var(--c-danger); color: var(--c-danger); background: rgba(255,0,0,0.1); }
    .btn-cyber.danger:hover { background: var(--c-danger); color: #000; box-shadow: 0 0 15px var(--c-danger); }

    /* Risk Ring */
    .risk-ring-wrapper { position: relative; width: 100px; height: 100px; margin: 0 auto; }
    .risk-ring { transform: rotate(-90deg); width: 100%; height: 100%; }
    .ring-bg { fill: none; stroke: #222; stroke-width: 8; }
    .ring-fill { fill: none; stroke: var(--c-accent); stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 0.5s, stroke 0.5s; }
    .risk-center { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .risk-pct { font-size: 20px; font-weight: bold; }
    .risk-label { font-size: 10px; color: #888; }

    /* Critical Overlay */
    .critical-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 0, 0, 0.2);
        z-index: 100;
        display: none;
        align-items: center; justify-content: center;
        backdrop-filter: blur(5px);
    }
    .critical-box {
        background: #000; border: 2px solid var(--c-danger);
        padding: 40px; text-align: center;
        box-shadow: 0 0 50px var(--c-danger);
        animation: shake 0.5s infinite;
    }
    .critical-title { color: var(--c-danger); font-size: 32px; font-weight: bold; margin-bottom: 10px; }
    @keyframes shake { 0% { transform: translate(1px, 1px); } 50% { transform: translate(-1px, -1px); } 100% { transform: translate(1px, 1px); } }

  </style>
</head>
<body>

  <div class="scanlines" aria-hidden="true"></div>

  <header class="header">
    <div class="header-left">
      <div class="logo-mark">
        <span class="logo-dot"></span><span class="logo-dot"></span><span class="logo-dot"></span>
      </div>
      <div>
        <h1 class="logo-text">CROWD<span>SENTINEL</span></h1>
        <p class="logo-sub">AI-IoT Stampede Risk Prediction System</p>
      </div>
    </div>
    <div class="header-right">
      <div class="live-badge" id="liveBadge" style="opacity: 0.3; filter: grayscale(1);">
        <span class="pulse-ring"></span>
        <span class="live-text">OFFLINE</span>
      </div>
    </div>
  </header>

  <main class="main-grid">

    <section class="panel panel--feed">
      <div class="panel-label">
        <span class="label-icon">◈</span> LIVE FEED — RTSP 10.76.11.62
      </div>
      <div class="feed-wrapper">
        <img id="videoFeed" src="" alt="System Standby" />
        <div class="feed-overlay">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
        </div>
      </div>
    </section>

    <aside class="sidebar">

      <div class="controls">
        <button id="btnStreamToggle" class="btn-cyber" onclick="toggleSystem()">
          INITIALIZE SYSTEM
        </button>
        <input type="file" id="imgUpload" accept="image/*" style="display: none" onchange="handleUpload(this)"/>
        <button class="btn-cyber" onclick="document.getElementById('imgUpload').click()">
          📂 ANALYZE STATIC IMG
        </button>
        <div id="uploadResult" style="font-size: 12px; color: #888; text-align: center; height: 15px;"></div>
      </div>

      <div class="card card--count">
        <div class="card__label">DETECTED PERSONS</div>
        <div class="count-display">
          <span class="count-number" id="personCount">0</span>
          <span class="count-unit">PPL</span>
        </div>
      </div>

      <div class="card card--risk">
        <div class="card__label">STAMPEDE RISK</div>
        <div class="risk-ring-wrapper">
          <svg class="risk-ring" viewBox="0 0 120 120">
            <circle class="ring-bg" cx="60" cy="60" r="50" />
            <circle class="ring-fill" id="riskRingFill" cx="60" cy="60" r="50" stroke-dasharray="314" stroke-dashoffset="314" />
          </svg>
          <div class="risk-center">
            <div class="risk-pct" id="riskPct">0%</div>
            <div class="risk-label" id="riskLabel">SAFE</div>
          </div>
        </div>
      </div>

    </aside>

  </main>

  <div class="critical-overlay" id="criticalOverlay">
    <div class="critical-box">
      <div class="critical-title">CRITICAL RISK</div>
      <div>DENSITY THRESHOLD EXCEEDED</div>
      <button class="btn-cyber danger" style="margin-top:20px" onclick="dismissCritical()">ACKNOWLEDGE</button>
    </div>
  </div>

  <script>
    let isRunning = false;
    let eventSource = null;
    const CRITICAL_THRESHOLD = 50; // Set Alert Threshold
    const MAX_CAPACITY = 100;      // For Ring Calculation

    async function toggleSystem() {
        const btn = document.getElementById('btnStreamToggle');
        const img = document.getElementById('videoFeed');
        const badge = document.getElementById('liveBadge');
        const badgeText = badge.querySelector('.live-text');

        if (!isRunning) {
            // Start
            btn.innerText = "TERMINATE STREAM";
            btn.classList.add("danger");
            
            const res = await fetch('/start_stream', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started' || data.status === 'already_running') {
                img.src = "/video_feed?" + new Date().getTime();
                img.classList.add("active");
                
                badge.style.opacity = "1";
                badge.style.filter = "none";
                badgeText.innerText = "ONLINE";

                startDataStream();
                isRunning = true;
            }
        } else {
            // Stop
            btn.innerText = "INITIALIZE SYSTEM";
            btn.classList.remove("danger");

            await fetch('/stop_stream', { method: 'POST' });
            
            img.classList.remove("active");
            setTimeout(() => { img.src = ""; }, 500);

            badge.style.opacity = "0.3";
            badge.style.filter = "grayscale(1)";
            badgeText.innerText = "OFFLINE";

            stopDataStream();
            isRunning = false;
        }
    }

    function startDataStream() {
        if (eventSource) eventSource.close();
        eventSource = new EventSource("/count_feed");
        
        eventSource.onmessage = (e) => {
            const count = parseInt(e.data);
            updateDashboard(count);
        };
    }

    function stopDataStream() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        updateDashboard(0);
    }

    function updateDashboard(count) {
        // Update Number
        document.getElementById("personCount").innerText = count;

        // Update Ring
        const circle = document.getElementById('riskRingFill');
        const radius = circle.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        
        const percent = Math.min(count / MAX_CAPACITY, 1);
        const offset = circumference - (percent * circumference);
        circle.style.strokeDashoffset = offset;

        // Risk Logic
        const pctText = Math.round(percent * 100) + "%";
        document.getElementById("riskPct").innerText = pctText;
        
        const riskLabel = document.getElementById("riskLabel");
        
        if (count > CRITICAL_THRESHOLD) {
            circle.style.stroke = "#ff003c";
            riskLabel.innerText = "CRITICAL";
            riskLabel.style.color = "#ff003c";
            document.getElementById("criticalOverlay").style.display = "flex";
        } else if (count > CRITICAL_THRESHOLD * 0.5) {
            circle.style.stroke = "#ffa500";
            riskLabel.innerText = "MODERATE";
            riskLabel.style.color = "#ffa500";
        } else {
            circle.style.stroke = "#00ffff";
            riskLabel.innerText = "SAFE";
            riskLabel.style.color = "#888";
        }
    }

    function dismissCritical() {
        document.getElementById("criticalOverlay").style.display = "none";
    }

    async function handleUpload(input) {
        if (input.files.length === 0) return;
        
        const formData = new FormData();
        formData.append('file', input.files[0]);
        
        document.getElementById('uploadResult').innerText = "Analyzing...";

        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        
        document.getElementById('uploadResult').innerText = 
            `Static Analysis Result: ${data.detected_count} Persons`;
    }
  </script>
</body>
</html>
    """)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)