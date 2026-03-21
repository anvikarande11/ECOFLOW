from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time
import os

app = Flask(__name__)

# --- 1. AI SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Forces CPU to prevent GPU memory lag

model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]

stats = {"dry": 0, "wet": 0}

# --- 2. THE GENERATOR ---
def generate_frames():
    global stats
    # CAP_DSHOW is vital for Windows camera stability
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set resolution to 640x480 for a crisp LinkedIn video
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_count_time = 0 
    last_spoken_label = "" 

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 1. AI Analysis (on a copy to keep the main frame clean)
            img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1
            
            prediction = model.predict(img, verbose=0)
            best_index = np.argmax(prediction)
            conf = prediction[0][best_index]
            current_label = class_names[best_index]
            current_time = time.time()

            # 2. Logic (Matching your main.py)
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
            
            if conf < 0.60:
                last_spoken_label = ""

            # 3. Visual Feedback (BGR Colors)
            # Dry = Green, Wet = Red
            color = (0, 255, 0) if "DRY" in current_label else (0, 0, 255) if "WET" in current_label else (255, 255, 255)
            
            # Adding a professional "Scanner" line or HUD effect
            cv2.putText(frame, f"STATUS: {current_label}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"CONFIDENCE: {int(conf*100)}%", (30, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 4. Stream Optimization (Quality=80 is great for web)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        # Ensures camera light turns off immediately
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
    # This wipes the dictionary back to zero
    stats = {"dry": 0, "wet": 0} 
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    # threaded=True allows the UI to update while the camera runs
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

@app.route('/reset_stats')
def reset_stats():
    global stats
    stats = {"dry": 0, "wet": 0}
    return jsonify({"status": "success", "stats": stats})