✋ Sign Language Digit Recognition Project<br>
📌 Description<br>
<br>
This project uses an image dataset of hand gesture digits representing numbers 0 to 9 in sign language format.<br>

Each class corresponds to a digit and contains grayscale or RGB images of hand signs captured under varying conditions. The project uses a Convolutional Neural Network (CNN) to classify the images into the correct digit class.<br>

🧠 Model & Approach<br>
<br>
Model Type: Convolutional Neural Network (CNN)<br>

Framework: TensorFlow / Keras<br>

Input: Images of hand gestures (0–9)<br>

Output: Predicted digit class<br>

Key Steps:<br>

Image preprocessing: resizing and normalization<br>

Dataset split: training & testing<br>

CNN architecture: convolution + pooling + dense layers<br>

Model training over multiple epochs<br>

Performance evaluation using accuracy and confusion matrix<br>

Focused on realistic performance and understandable model pipeline rather than overfitting.<br><br>

📂 Dataset Structure<br>
data/<br>
├── 0/<br>
├── 1/<br>
├── 2/<br>
├── 3/<br>
├── 4/<br>
├── 5/<br>
├── 6/<br>
├── 7/<br>
├── 8/<br>
└── 9/<br>

<br>
Each folder contains images corresponding to the digit label.<br>
<br>
📥 Dataset Source<br>
<br>
The dataset can be either:<br>

Collected manually using a camera/webcam<br>
OR<br>

Sourced from a publicly available sign language digit dataset<br>

Exact link of dataset - https://www.kaggle.com/datasets/pranavsharma1670/sign-language-recognition-dataset <br>
<br>
🚫 Why dataset is not included<br>
<br>
The dataset is not uploaded to GitHub to:<br>

Avoid large repository size<br>

Respect dataset licensing (if applicable)<br>

Follow good version control practices<br>
<br><br>
🔁 How to recreate the dataset<br>
<br>
Obtain or create a sign language digit dataset (0–9)<br>

Place the images in the folder structure shown above<br>

Ensure image dimensions match the preprocessing step in the code<br>

Once the dataset is placed correctly, the training and testing scripts will run as expected.<br>

<br>

📊 Results<br>
<br>
Model achieves good classification performance on most digit classes<br>

Confusion matrix highlights digits that are visually similar and harder to predict<br>

Focus is on accuracy that is realistic and replicable<br>
<br><br>
🚀 Future Improvements<br><br>

Improve recognition of visually confusing digits<br>

Use larger and more diverse datasets<br>

Add real-time webcam prediction<br>

Experiment with deeper CNN architectures<br>

Extend from digits to full sign language alphabet<br><br>

🧑‍💻 Author<br><br>

Chittransh<br>
B.Tech Student | Machine Learning & AI Enthusiast<br>
