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

# Load model once when server starts
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]

# Global Stats
stats = {"dry": 0, "wet": 0}
last_count_time = 0
last_spoken_label = ""

# --- 2. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global stats, last_count_time, last_spoken_label
    try:
        # Get image from phone
        data = request.json['image']
        encoded_data = data.split(',')[1] if ',' in data else data
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # AI Processing
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

        return jsonify({
            "status": "success",
            "label": current_label,
            "conf": int(conf * 100),
            "stats": stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_stats')
def get_stats():
    return jsonify(stats)

@app.route('/reset_stats')
def reset_stats():
    global stats, last_count_time, last_spoken_label
    stats = {"dry": 0, "wet": 0} 
    last_count_time = 0
    last_spoken_label = ""
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)