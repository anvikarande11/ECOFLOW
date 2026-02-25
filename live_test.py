import cv2
import numpy as np
import tensorflow as tf

# 1. Load the brain you just created
model = tf.keras.models.load_model('ecoflow_model.h5')

# 2. These must match the order of your folders (usually alphabetical)
# If your folders were 'dry_waste' and 'wet_waste', the list is:
labels = ['Dry Waste', 'Wet Waste']

cap = cv2.VideoCapture(0)

print("EcoFlow is live! Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Prepare the image for the model
    # The model expects a 224x224 image, normalized to 0-1
    img = cv2.resize(frame, (224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0            # Normalize

    # 4. Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get index of highest probability
    result_index = np.argmax(predictions[0])
    label = labels[result_index]
    confidence = predictions[0][result_index] * 100

    # 5. Visual Feedback
    # Green for Wet, Blue for Dry
    color = (0, 255, 0) if "Wet" in label else (255, 0, 0)
    
    cv2.rectangle(frame, (0, 0), (450, 80), (255, 255, 255), -1) # White background for text
    cv2.putText(frame, f"{label}: {confidence:.1f}%", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('EcoFlow Identifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()