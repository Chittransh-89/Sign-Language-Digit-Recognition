# Sign-Language-Digit-Recognition <br>
📌 Overview<br>

-> This project focuses on recognizing hand sign digits (0–9) from images using Convolutional Neural Networks (CNNs).<br>
-> The goal is to build a machine learning system that can accurately classify sign language digits, which can be extended towards assistive technologies for the hearing-impaired.<br>
-> Model is trained on a labeled image dataset of hand gestures and is capable of making predictions on unseen images and real-time inputs.<br>

🧠 Model & Approach<br>

-> Model Type: Convolutional Neural Network (CNN)<br>
-> Framework: TensorFlow / Keras<br>
-> Input: Hand gesture images representing digits (0–9)<br>
-> Output: Predicted digit class<br>

Key Steps:<br>

-> Image resizing and normalization<br>
-> Dataset splitting (training & testing)<br>
-> CNN model design with convolution, pooling, and dense layers<br>
-> Model training over multiple epochs<br>
-> Performance evaluation using accuracy and confusion matrix<br>

The focus was on building a stable and understandable CNN pipeline, not blindly increasing complexity.<br>

📂 Project Structure<br>
Sign-Language-Digit-Recognition/<br>
│<br>
├── dataset/                 # Image dataset (0–9 folders)<br>
├── src/<br>
│   ├── new_training.py             # Model training script<br>
│   ├── new_testing.py              # Model testing & evaluation<br>
│   ├── realtime_test.py           # Prediction on new images<br>
│<br>
├── requirements.txt         # Required Python libraries<br>
├── README.md                # Project documentation<br>

⚙️ How to Run<br>

-> Follow these steps to run the project locally:<br>
Clone the repository<br>

git clone https://github.com/Chittransh-89/Sign-Language-Digit-Recognition.git<br>
cd sign-language-digit-recognition<br>


-> Install dependencies<br>
pip install -r requirements.txt<br>


-> Train the model<br>
python src/new_training.py<br>

-> Test / Predict<br>
python src/new_testing.py<br>

📊 Results<br>

-> The model achieves good classification performance on most digit classes.<br>
-> A confusion matrix is used to analyze correct and incorrect predictions.<br>
-> Some digits with visually similar hand shapes require extra effort to predict accurately.<br>
-> The project prioritizes realistic performance over fake 99% accuracy claims.<br>

🚀 Future Improvements<br>

-> Improve detection of confusing digits (e.g., 8, 9, 4)<br>
-> Use larger and more diverse datasets<br>
-> Implement real-time webcam detection<br>
-> Experiment with deeper CNN architectures<br>
-> Extend from digits to full sign language alphabets<br>

🧑‍💻 Author<br>

Chittransh<br>
B.Tech Student | Machine Learning Enthusias<br>
Focused on practical AI projects and real-world problem solving.<br>
