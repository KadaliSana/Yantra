import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__, template_folder='.')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────
lock = threading.Lock()
outputFrame = None
current_count = 0

# Mode control: 'webcam' or 'image'
input_mode = 'webcam' 
uploaded_image_processed = None # Stores the processed static image

# ─── IMAGE PROCESSING ENGINE ─────────────────────────────────────────────────
def process_frame(frame):
    """
    Takes a raw frame (numpy array), detects faces, draws boxes,
    and returns the processed frame (JPEG bytes) and the count.
    """
    # Resize for consistency and performance
    height, width = frame.shape[:2]
    max_width = 800
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    count = len(faces)

    # Draw UI
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, "DETECTED", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        
    # Overlay "Source" indicator
    label = "LIVE FEED" if input_mode == 'webcam' else "STATIC IMAGE ANALYSIS"
    color = (0, 255, 0) if input_mode == 'webcam' else (0, 165, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Encode to JPEG
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    if not flag:
        return None, 0
        
    return encodedImage.tobytes(), count

# ─── WEBCAM THREAD ───────────────────────────────────────────────────────────
def capture_feed():
    global outputFrame, current_count
    cap = cv2.VideoCapture(0)
    time.sleep(2.0) # Warmup

    while True:
        # If we are in 'image' mode, we pause the webcam logic to save resources
        if input_mode == 'image':
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        jpeg_bytes, count = process_frame(frame)

        with lock:
            outputFrame = jpeg_bytes
            current_count = count
        
        time.sleep(0.03)

# ─── FLASK ROUTES ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/count_feed')
def count_feed():
    return Response(generate_sse(), mimetype="text/event-stream")

# NEW: Upload an image to override the webcam
@app.route('/upload', methods=['POST'])
def upload_image():
    global input_mode, outputFrame, current_count, uploaded_image_processed
    
    file = request.files['image']
    if not file:
        return jsonify({"error": "No file"}), 400

    # Convert uploaded file to numpy array for OpenCV
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Process immediately
    jpeg_bytes, count = process_frame(frame)

    with lock:
        input_mode = 'image'
        outputFrame = jpeg_bytes
        current_count = count
        uploaded_image_processed = jpeg_bytes # Cache it

    return jsonify({"status": "switched_to_image", "count": count})

# NEW: Reset back to webcam
@app.route('/reset', methods=['POST'])
def reset_feed():
    global input_mode
    with lock:
        input_mode = 'webcam'
    return jsonify({"status": "switched_to_webcam"})

# ─── GENERATORS ──────────────────────────────────────────────────────────────
def generate_mjpeg():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None:
                continue
            data = outputFrame
        
        # Standard MJPEG yield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        
        # If static image, we slow down the refresh rate significantly
        # to save bandwidth, since the image isn't changing.
        delay = 0.05 if input_mode == 'webcam' else 1.0
        time.sleep(delay)

def generate_sse():
    global current_count
    last_sent = -1
    while True:
        with lock:
            cnt = current_count
        
        # Always send heartbeat every second, or immediately on change
        yield f"data: {cnt}\n\n"
        time.sleep(0.5)

if __name__ == '__main__':
    t = threading.Thread(target=capture_feed, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)