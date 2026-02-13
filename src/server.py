import cv2
import boto3
import time
import threading
import queue
import os
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()  # This loads the variables from .env

app = Flask(__name__)

# --- Configuration ---
# Now fetching from the .env file
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
STREAM_NAME = os.getenv("STREAM_NAME")

# --- Shared State ---
class VideoState:
    def __init__(self):
        self.frame = None
        self.count = 0
        self.lock = threading.Lock()
        self.running = True

state = VideoState()

# --- Thread 1: The "Buffer Filler" (Reads KVS) ---
def capture_thread():
    print("[Thread 1] Connecting to AWS KVS...")
    
    # Check if keys loaded correctly
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("❌ Error: AWS credentials not found. Check your .env file.")
        return

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY, 
        aws_secret_access_key=AWS_SECRET_KEY, 
        region_name=AWS_REGION
    )
    kvs = session.client('kinesisvideo')
    
    try:
        # Get the HLS URL
        endpoint = kvs.get_data_endpoint(StreamName=STREAM_NAME, APIName='GET_HLS_STREAMING_SESSION_URL')['DataEndpoint']
        kvam = session.client('kinesis-video-archived-media', endpoint_url=endpoint)
        url = kvam.get_hls_streaming_session_url(
            StreamName=STREAM_NAME,
            PlaybackMode='LIVE',
            HLSFragmentSelector={'FragmentSelectorType': 'SERVER_TIMESTAMP'},
            ContainerFormat='MPEG_TS',
            DiscontinuityMode='ON_DISCONTINUITY',
            DisplayFragmentTimestamp='NEVER'
        )['HLSStreamingSessionURL']

        # Open the stream
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # ⚠️ CRITICAL: Limit internal buffer to 1 frame to prevent "ghost" lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while state.running:
            ret, frame = cap.read()
            if not ret:
                print("Stream interrupted. Reconnecting...")
                time.sleep(2)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            
            # Update the shared buffer immediately
            with state.lock:
                state.frame = frame
            
            # We don't sleep here; we want to read as fast as KVS sends data.
    except Exception as e:
        print(f"Error in capture thread: {e}")

# --- Thread 2: The "AI Worker" (Reads Buffer -> Processes) ---
def ai_thread():
    print("[Thread 2] Loading YOLO...")
    try:
        model = YOLO("best.pt")
        
        while state.running:
            # 1. Grab a snapshot of the current frame
            with state.lock:
                if state.frame is None:
                    continue
                # Copy effectively creates a new buffer for the AI to work on
                # while the main thread continues updating the real frame.
                working_frame = state.frame.copy()

            # 2. Run Inference (This might take 100ms - 500ms)
            # We resize to 640 for speed, detection accuracy usually remains high
            input_frame = cv2.resize(working_frame, (640, 360))
            results = model(input_frame, conf=0.5, classes=[0], verbose=False)
            
            # 3. Update the count
            with state.lock:
                state.count = len(results[0].boxes)
            
            # AI runs at its own max speed. It effectively "skips" frames
            # that appeared while it was processing this one.
    except Exception as e:
        print(f"Error in AI thread: {e}")

# Start Threads
t1 = threading.Thread(target=capture_thread, daemon=True).start()
t2 = threading.Thread(target=ai_thread, daemon=True).start()

# --- Flask: The "Viewer" (Reads Buffer -> Browser) ---
def generate_mjpeg():
    while True:
        with state.lock:
            if state.frame is None:
                # If no frame yet, yield a placeholder or wait slightly
                time.sleep(0.1)
                continue
            # Encode the LATEST frame available
            ret, buffer = cv2.imencode('.jpg', state.frame)

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Cap sending rate to ~30 FPS to save bandwidth
        time.sleep(0.03)

def generate_sse_count():
    last_count = -1
    while True:
        with state.lock:
            c = state.count
        
        if c != last_count:
            yield f"data: {c}\n\n"
            last_count = c
        time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_feed')
def count_feed():
    return Response(generate_sse_count(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <body style="font-family: sans-serif; text-align: center; background: #222; color: #fff;">
                <h1>Zero-Lag Monitor</h1>
                <img src="/video_feed" style="width: 80%; border: 2px solid #555;"/><br/>
                <h2 style="font-size: 50px; color: #0f0;">
                    People: <span id="cnt">0</span>
                </h2>
                <script>
                    new EventSource("/count_feed").onmessage = (e) => {
                        document.getElementById("cnt").innerText = e.data;
                    };
                </script>
            </body>
        </html>
    """)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
