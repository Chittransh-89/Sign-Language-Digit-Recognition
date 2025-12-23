import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ===== CONFIG =====
MODEL_PATH = "new_training.keras"
IMG_SIZE = 128 # Use the same size as training
ROI_X1, ROI_Y1 = 300, 100
ROI_X2, ROI_Y2 = 600, 400
SMOOTH_FRAMES = 15  # number of frames to smooth prediction

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)
print("Model loaded")

# ===== OPEN CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opening")
    exit()

# Queue to store last predictions
pred_queue = deque(maxlen=SMOOTH_FRAMES)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw ROI box
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand inside box",
                (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Crop ROI
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    if roi.size != 0:
        # Preprocess
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)
        pred_queue.append(prediction[0])  # store probabilities

        # Average predictions over last SMOOTH_FRAMES frames
        avg_pred = np.mean(pred_queue, axis=0)
        digit = np.argmax(avg_pred)
        confidence = np.max(avg_pred) * 100

        # Display prediction
        cv2.putText(frame,
                    f"Digit: {digit} ({confidence:.1f}%)",
                    (ROI_X1, ROI_Y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

    cv2.imshow("Sign Digit Recognition - Real Time", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
