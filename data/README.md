# Sign-Language-Digit-Recognition 
📌 Overview

-> This project focuses on recognizing hand sign digits (0–9) from images using Convolutional Neural Networks (CNNs).
-> The goal is to build a machine learning system that can accurately classify sign language digits, which can be extended towards assistive technologies for the hearing-impaired.
-> Model is trained on a labeled image dataset of hand gestures and is capable of making predictions on unseen images and real-time inputs.

🧠 Model & Approach

-> Model Type: Convolutional Neural Network (CNN)
-> Framework: TensorFlow / Keras
-> Input: Hand gesture images representing digits (0–9)
-> Output: Predicted digit class

Key Steps:

-> Image resizing and normalization
-> Dataset splitting (training & testing)
-> CNN model design with convolution, pooling, and dense layers
-> Model training over multiple epochs
-> Performance evaluation using accuracy and confusion matrix

The focus was on building a stable and understandable CNN pipeline, not blindly increasing complexity.

📂 Project Structure
Sign-Language-Digit-Recognition/
│
├── dataset/                 # Image dataset (0–9 folders)
├── src/
│   ├── new_training.py             # Model training script
│   ├── new_testing.py              # Model testing & evaluation
│   ├── realtime_test.py           # Prediction on new images
│
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation

⚙️ How to Run

-> Follow these steps to run the project locally:
Clone the repository

git clone https://github.com/your-username/sign-language-digit-recognition.git
cd sign-language-digit-recognition


-> Install dependencies
pip install -r requirements.txt


-> Train the model
python src/new_training.py

-> Test / Predict
python src/new_testing.py

📊 Results

-> The model achieves good classification performance on most digit classes.
-> A confusion matrix is used to analyze correct and incorrect predictions.
-> Some digits with visually similar hand shapes require extra effort to predict accurately.
-> The project prioritizes realistic performance over fake 99% accuracy claims.

🚀 Future Improvements

-> Improve detection of confusing digits (e.g., 8, 9, 4)
-> Use larger and more diverse datasets
-> Implement real-time webcam detection
-> Experiment with deeper CNN architectures
-> Extend from digits to full sign language alphabets

🧑‍💻 Author

Chittransh
B.Tech Student | Machine Learning Enthusias
Focused on practical AI projects and real-world problem solving.
