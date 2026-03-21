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

# Global stats and label tracking
stats = {"dry": 0, "wet": 0}
current_label = "INITIALIZING..." # Defined globally so all routes can see it

def generate_frames():
    global stats, current_label
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Internal tracking for efficiency (Matches your main.py)
    last_count_time = 0 
    last_spoken_label = "" 

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
        
        # UPDATE THE GLOBAL LABEL
        current_label = class_names[best_index].strip()
        current_time = time.time()

        # Efficiency Logic: Only count if high confidence and time has passed
        if conf > 0.90 and (current_time - last_count_time) > 2.5:
            if current_label != last_spoken_label:
                if "WET" in current_label:
                    stats["wet"] += 1
                elif "DRY" in current_label:
                    stats["dry"] += 1
                
                last_count_time = current_time
                last_spoken_label = current_label
        
        # Reset tracker if screen is empty
        if conf < 0.60:
            last_spoken_label = ""

        # Visual HUD
        color = (0, 255, 0) if "DRY" in current_label else (0, 0, 255) if "WET" in current_label else (255, 255, 255)
        cv2.putText(frame, f"SIGHT: {current_label} ({int(conf*100)}%)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# --- 2. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    # Including the label in the stats JSON for the frontend to use
    return jsonify({
        "dry": stats["dry"], 
        "wet": stats["wet"], 
        "label": current_label 
    })

@app.route('/reset_stats')
def reset_stats():
    global stats
    stats = {"dry": 0, "wet": 0}
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    # Added links to terminal as requested
    print("\n" + "="*50)
    print("ECOFLOW ACTIVE: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)