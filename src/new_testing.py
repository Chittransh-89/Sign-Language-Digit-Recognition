import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
DATASET_PATH = "New_Dataset"
MODEL_PATH = "new_training.keras"
IMG_SIZE = 128

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)
print("Model loaded")

# ===== LOAD DATASET (same as training) =====
labels = sorted(os.listdir(DATASET_PATH))

X = []
Y = []

for label in labels:
    class_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            X.append(img)
            Y.append(int(label))

X = np.array(X)
Y = np.array(Y)

print("Dataset loaded")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ===== SAME SPLIT AS TRAINING =====
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ===== EVALUATION =====
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# ===== VISUAL TESTING (IMPORTANT) =====
for i in range(5):
    idx = np.random.randint(0, len(X_test))

    img = X_test[idx]
    true_label = Y_test[idx]

    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_label = np.argmax(pred)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label} | Predicted: {pred_label}")
    plt.show()
