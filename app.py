from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import base64
import time
import os

app = Flask(__name__)

# --- 1. AI SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]

stats = {"dry": 0, "wet": 0}
last_count_time = 0 
last_spoken_label = "" 

# Helper function to process AI logic (Shared by Local and Cloud)
def predict_logic(frame):
    global stats, last_count_time, last_spoken_label
    
    # AI Analysis
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1
    
    prediction = model.predict(img, verbose=0)
    best_index = np.argmax(prediction)
    conf = float(prediction[0][best_index])
    current_label = class_names[best_index].strip()
    current_time = time.time()

    # Logic: Count objects
    if conf > 0.90 and (current_time - last_count_time) > 2.5:
        if current_label != last_spoken_label:
            if "WET" in current_label:
                stats["wet"] += 1
                last_count_time = current_time
                last_spoken_label = current_label
            elif "DRY" in current_label:
                stats["dry"] += 1
                last_count_time = current_time
                last_spoken_label = current_label
    
    if conf < 0.50:
        last_spoken_label = ""
        
    return current_label, conf

# --- 2. THE GENERATOR (For Local PC Test) ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        label, conf = predict_logic(frame)
        # Draw on frame for local preview
        cv2.putText(frame, f"{label} {int(conf*100)}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- 3. THE 4 ESSENTIAL ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

# LINK 1: The Cloud Processor (REQUIRED FOR PHONE/RENDER)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        encoded_data = data.split(',')[1] if ',' in data else data
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        label, conf = predict_logic(frame)
        
        return jsonify({
            "status": "success",
            "label": label,
            "conf": int(conf * 100),
            "stats": stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# LINK 2: Live Video Stream (For Local PC only)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# LINK 3: Get Stats (To update the numbers on your website)
@app.route('/get_stats')
def get_stats():
    return jsonify(stats)

# LINK 4: Reset
@app.route('/reset_stats')
def reset_stats():
    global stats, last_spoken_label
    stats = {"dry": 0, "wet": 0} 
    last_spoken_label = ""
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    # Use Render's port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)