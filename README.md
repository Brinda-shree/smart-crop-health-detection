# 🌿 Smart Crop Health Detection System

An AI-powered system that automatically detects crop diseases from leaf images using deep learning. Built using **MobileNetV2**, this system enables fast and accurate health analysis of crops, aiding farmers in early diagnosis and management.

---

## 👩‍💻 Developed By

**S. Brinda Shree**  
Solo Participant, SA Engineering College

---

## 🚀 Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Scikit-learn
- Matplotlib & Seaborn
- MobileNetV2 (Pre-trained CNN)
- Google Colab / Jupyter Notebook
- Git & GitHub

---

## 🧠 Features

- Detects **plant diseases** using deep learning
- Built with **MobileNetV2** for fast performance
- Trained on labeled crop images with high accuracy
- Saves prediction reports with confidence scores
- Generates performance plots (accuracy/loss graphs)
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

---

## 📁 Project Structure

crop-health-monitor/
│
├── data/ # Dataset (train/test)
├── model/ # Saved model and training script
│ ├── train.py
│ └── model.h5
├── predict.py # Script to make predictions on new images
├── reports/ # Evaluation metrics and plots
├── utils/ # (Optional) Helper functions
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 📸 Sample Output

- ✅ Predicted Class: **Tomato___Leaf_Mold**
- 🧪 Confidence: **99.87%**
- 📊 Evaluation Accuracy: **98.75%**

---

## 📈 Model Evaluation

- **Train Accuracy**: 98.75%
- **Validation Accuracy**: 99.78%
- **Loss**: Reduced steadily over 15 epochs
- Confusion matrix and classification report available in `reports/`

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Brinda-shree/smart-crop-health-detection.git
cd smart-crop-health-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python model/train.py

# 4. Predict using an image
python predict.py --image "path_to_leaf.jpg"


📄 License
This project is for educational and hackathon use only. Contact the author for other use cases.

🌱 Acknowledgements
PlantVillage Dataset

TensorFlow & Keras Documentation
