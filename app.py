from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
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

# --- 2. THE GENERATOR ---
def generate_frames():
    global stats
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_count_time = 0 
    last_spoken_label = "" 

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # AI Analysis
            img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1
            
            prediction = model.predict(img, verbose=0)
            best_index = np.argmax(prediction)
            conf = prediction[0][best_index]
            current_label = class_names[best_index]
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

            # DRAWING HUD ON VIDEO
            color = (0, 255, 0) if "DRY" in current_label else (0, 0, 255) if "WET" in current_label else (255, 255, 255)
            cv2.putText(frame, f"IDENTIFIED: {current_label}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"CONFIDENCE: {int(conf*100)}%", (30, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    return jsonify(stats)

@app.route('/reset_stats')
def reset_stats():
    global stats
    stats = {"dry": 0, "wet": 0} 
    return jsonify({"status": "success", "stats": stats})

# --- NEW LINKS ---

@app.route('/export_data')
def export_data():
    """Download current stats as a JSON file."""
    import json
    from flask import make_response
    
    # Create a simple report
    report = {
        "project": "EcoFlow AI",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": stats,
        "total_items": stats["dry"] + stats["wet"]
    }
    
    response = make_response(json.dumps(report, indent=4))
    response.headers.set('Content-Type', 'application/json')
    response.headers.set('Content-Disposition', 'attachment', filename='ecoflow_report.json')
    return response

if __name__ == '__main__':
    # '0.0.0.0' allows external devices (like your phone) to connect
    # os.environ.get('PORT') allows Render to tell the app which port to use
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)