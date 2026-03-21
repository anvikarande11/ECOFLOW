import os
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# 1. Voice Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 2. Model Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip().upper() for line in f.readlines()]

dry_count = 0
wet_count = 0
last_count_time = 0 
last_spoken_label = "" # Keeps track of what was just said

cap = cv2.VideoCapture(0)

print(f"System ready. Labels: {class_names}")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Pre-process image
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1

    # Predict
    prediction = model.predict(img, verbose=0)
    best_index = np.argmax(prediction)
    conf = prediction[0][best_index]
    current_label = class_names[best_index]

    current_time = time.time()
    
    # 3. BALANCED TRIGGER LOGIC
    # We allow a new voice alert if:
    # A) 3 seconds have passed since the last one
    # B) The label has CHANGED (e.g., from Wet to Dry)
    
    if conf > 0.90 and (current_time - last_count_time) > 2.5:
        
        # Check if we are seeing a DIFFERENT item than the last one spoken
        if current_label != last_spoken_label:
            
            if "WET" in current_label:
                wet_count += 1
                print(">>> VOICE: WET WASTE")
                engine.say("Wet waste detected")
                engine.runAndWait()
                last_count_time = current_time
                last_spoken_label = current_label
                
            elif "DRY" in current_label:
                dry_count += 1
                print(">>> VOICE: DRY WASTE")
                engine.say("Dry waste detected")
                engine.runAndWait()
                last_count_time = current_time
                last_spoken_label = current_label

    # 4. RESET THE "LAST SPOKEN"
    # If confidence is low, it means the screen is empty. 
    # We reset last_spoken_label so it can detect the SAME type of item again later.
    if conf < 0.60:
        last_spoken_label = ""

    # 5. UI
    cv2.rectangle(frame, (0, 0), (250, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"DRY: {dry_count}", (10, 30), 1, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, f"WET: {wet_count}", (10, 65), 1, 1.5, (0, 0, 255), 2)
    
    # Show what the AI is currently seeing
    status_color = (0, 255, 0) if "DRY" in current_label else (0, 0, 255)
    cv2.putText(frame, f"SIGHT: {current_label} ({int(conf*100)}%)", (10, 450), 1, 1.5, status_color, 2)

    cv2.imshow("EcoFlow Balanced Sorter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()