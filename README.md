Indian Sign Language Detection using MediaPipe

This project implements a real-time Indian Sign Language (ISL) detection system using MediaPipe for hand landmark extraction and a Machine Learning classifier for gesture recognition. The system captures live video from a webcam, extracts keypoints from hand movements, and predicts ISL gestures instantly.

🚀 Features

Real-time hand tracking using MediaPipe Holistic
Keypoint extraction for hands, pose, and face
Dataset generation scripts for custom gestures
Sequence-based ML model for gesture classification
Live prediction with webcam input
User-friendly and extendable project structure

📁 Project Structure
├── dataset_keypoint_generation.py   # Script to create training dataset
├── isl_detection.py                 # Real-time detection script
├── keypoint.csv                     # Example extracted keypoints
├── ISL_classifier.ipynb             # Model training notebook
├── mediapipe_utils/                 # Utility functions (if present)
└── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/yourusername/Jyothsna-Sign-Language-Detection-using-Mediapipe
cd Jyothsna-Sign-Language-Detection-using-Mediapipe

2. Install dependencies
pip install -r requirements.txt

▶️ How to Run
Dataset Generation
python dataset_keypoint_generation.py

Real-time Detection
python isl_detection.py

Make sure your webcam is connected.

🧠 Model Training

Open the Jupyter Notebook:
jupyter notebook ISL_classifier.ipynb
Train your custom ISL gesture classifier.

🙌 Technologies Used

Python
MediaPipe
OpenCV
NumPy / Pandas
Scikit-learn / TensorFlow

🎯 Future Enhancements

Support more ISL gestures
Build a GUI interface
Deploy as a web app
Add sentence-level ISL recognition

❤️ Contributors
H. Jyothsna
