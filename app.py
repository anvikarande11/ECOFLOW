from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
import time
import os

app = Flask(__name__)

# --- AI MODEL SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]

# Global Variables for Logic
stats = {"dry": 0, "wet": 0}
last_count_time = 0
last_label = ""
current_label = "AWAITING DATA"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global stats, last_count_time, last_label, current_label
    try:
        # 1. Decode Image from Browser
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. AI Prediction
        img = cv2.resize(frame, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1
        
        prediction = model.predict(img, verbose=0)
        best_index = np.argmax(prediction)
        conf = prediction[0][best_index]
        current_label = class_names[best_index].strip()
        current_time = time.time()

        # 3. Balanced Trigger Logic (From your original main.py)
        if conf > 0.90 and (current_time - last_count_time) > 2.5:
            if current_label != last_label:
                if "WET" in current_label:
                    stats["wet"] += 1
                    last_count_time = current_time
                    last_label = current_label
                elif "DRY" in current_label:
                    stats["dry"] += 1
                    last_count_time = current_time
                    last_label = current_label
        
        # Reset if screen is empty
        if conf < 0.60:
            last_label = ""

        return jsonify({
            "status": "success", 
            "label": current_label, 
            "stats": stats,
            "conf": int(conf * 100)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/reset_stats')
def reset_stats():
    global stats, last_label
    stats = {"dry": 0, "wet": 0}
    last_label = ""
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    # host 0.0.0.0 is required for Render/Cloud
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)