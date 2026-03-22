from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
import time
import os

app = Flask(__name__)

# --- AI MODEL SETUP ---
# Prevent TensorFlow from flooding the logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load your specific Teachable Machine model
try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]
except Exception as e:
    print(f"Error loading model or labels: {e}")

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
        # 1. Get JSON data from request
        json_data = request.get_json()
        if not json_data or 'image' not in json_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        data = json_data['image']
        
        # 2. Extract and Decode Base64
        # Browser sends: "data:image/jpeg;base64,/9j/4AAQ..."
        if ',' in data:
            encoded_data = data.split(',')[1]
        else:
            encoded_data = data
            
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400

        # 3. AI Prediction Pre-processing
        img = cv2.resize(frame, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1 # Normalize to [-1, 1] for Teachable Machine
        
        prediction = model.predict(img, verbose=0)
        best_index = np.argmax(prediction)
        conf = float(prediction[0][best_index]) # Convert to float for JSON stability
        current_label = class_names[best_index].strip()
        current_time = time.time()

        # 4. Debounce Logic (Prevents double counting)
        # Only count if confidence > 90% and 2.5 seconds have passed since last count
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
        
        # Reset current label tracker if confidence drops (object removed)
        if conf < 0.60:
            last_label = ""

        return jsonify({
            "status": "success", 
            "label": current_label, 
            "stats": stats,
            "conf": int(conf * 100)
        })
    except Exception as e:
        # This will print the exact error in your Render logs
        print(f"SERVER ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset_stats')
def reset_stats():
    global stats, last_label, last_count_time
    stats = {"dry": 0, "wet": 0}
    last_label = ""
    last_count_time = 0
    return jsonify({"status": "success", "stats": stats})

if __name__ == '__main__':
    # host 0.0.0.0 and dynamic port are mandatory for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)