import os 
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = f"New_Dataset"
labels = sorted(os.listdir(DATASET_PATH))
print(labels)

X = []
Y = []

for label in labels:
    class_path = os.path.join(DATASET_PATH,label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path,img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img,(128,128))
            img = img / 255.0
            X.append(img)
            Y.append(int(label))

X = np.array(X)
Y = np.array(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range = 15,
    zoom_range = 0.2,
    height_shift_range = 0.1,
    width_shift_range = 0.1,
    horizontal_flip = True
)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=64),
    epochs=60,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)

# Save
model.save("new_training.keras")
print("Model trained & saved")

# Check shapes
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)

import matplotlib.pyplot as plt
plt.imshow(X_train[0])
plt.axis("off")
plt.show()
print("Label:", Y_train[0])




    