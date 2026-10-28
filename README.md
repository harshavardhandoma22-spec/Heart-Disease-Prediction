# Machine Learning for Real-Time Heart Disease Prediction

## Project Overview
This project is a Python-based desktop application that uses machine learning to classify heart-related ECG data into different categories.

The application uses an XGBoost classifier and provides a graphical interface built with Tkinter.

## Technologies Used
- Python
- Tkinter
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn

## Features
- Upload an ECG dataset in CSV format
- Preprocess features and labels
- Split data into training and validation sets
- Train an XGBoost classification model
- Display accuracy, precision, recall, and macro F1-score
- Visualize the confusion matrix
- Predict classes for records in a test CSV file

## Project Structure
```text
HeartDisease/
├── Main.py
├── run.bat
├── requirements.txt
├── README.md
└── Dataset/       # Keep local until source and license are verified
```

## Setup
1. Install Python 3.
2. Open a terminal in the project folder.
3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python Main.py
   ```

Alternatively, run `run.bat` on Windows.

## Important Note
This project is for educational purposes and is not intended to provide medical diagnoses or replace advice from qualified healthcare professionals.

## Dataset
The dataset source and usage license still need to be verified. The dataset files are therefore not included in this public repository yet.
