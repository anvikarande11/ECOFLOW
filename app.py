from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
import os

app = Flask(__name__)

# AI Setup
model = tf.keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[-1].upper() for line in f]

stats = {"dry": 0, "wet": 0}
current_label = "AWAITING DATA"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global stats, current_label
    try:
        # 1. Receive the image from the browser
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. AI Prediction Logic
        img = cv2.resize(frame, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1
        
        prediction = model.predict(img, verbose=0)
        best_index = np.argmax(prediction)
        conf = prediction[0][best_index]
        current_label = class_names[best_index]

        # 3. Counting Logic (Matches your main.py efficiency)
        if conf > 0.90:
            if "WET" in current_label:
                stats["wet"] += 1
            elif "DRY" in current_label:
                stats["dry"] += 1

        return jsonify({"status": "success", "label": current_label, "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)