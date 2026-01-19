# 🦈 Multiclass Fish Image Classification System
## 📌 Project Overview

This project focuses on multiclass image classification of fish species using Deep Learning.
Multiple convolutional neural network (CNN) architectures were trained and evaluated to identify the best-performing model. The final model was deployed as an interactive Streamlit web application that allows users to upload fish images and receive predictions with confidence scores.

## 🎯 Objectives

Preprocess and augment fish image data

Train a CNN model from scratch

Experiment with five pre-trained deep learning models

Fine-tune pre-trained models for better performance

Evaluate and compare all models using standard metrics

Deploy the best-performing model using Streamlit

## 🗂 Dataset Structure

dataset/

├── train/

│   ├── Class_1/

│   ├── Class_2/

│   └── ...

├── val/

│   ├── Class_1/

│   ├── Class_2/

│   └── ...

└── test/

│   ├── Class_1/

│   ├── Class_2/

│   └── ...


Train: Used for model training with augmentation

Validation: Used during training to monitor performance

Test: Used only for final evaluation

## ⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

Streamlit

Pillow (PIL)

## 🔄 Data Preprocessing & Augmentation

Pixel values rescaled to [0, 1]

Data augmentation techniques applied:

Rotation

Zoom

Horizontal flipping

Ensured better generalization and reduced overfitting

## 🧠 Model Training
### 1️⃣ CNN From Scratch

A custom CNN architecture was built using:

Convolutional layers

Max pooling layers

Fully connected layers

This model served as a baseline for comparison.

### 2️⃣ Pre-trained Models (Transfer Learning)

The following five pre-trained models were trained :

VGG16

ResNet50

MobileNetV2

InceptionV3

EfficientNetB0

### 📊 Model Evaluation

All models were evaluated on the test dataset using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Training history (accuracy & loss curves) was visualized for each model to analyze convergence and overfitting.

### 🏆 Best Model Selection

InceptionV3 achieved the highest test accuracy along with strong precision, recall, and F1-score.
It showed stable training behavior and minimal overfitting.

➡ Selected as the final deployment model

Saved as:

models/InceptionV3_best.h5

## 🚀 Deployment (Streamlit Application)
Features:

Upload a fish image (.jpg, .png, .jpeg)

Predict the fish category

Display prediction confidence score

## 📝 Conclusion

This project demonstrates an end-to-end deep learning pipeline, from data preprocessing and model training to evaluation and deployment. By comparing multiple CNN architectures, the most effective model was selected and deployed as a real-time web application.

## 📌 Future Enhancements

Add more fish species

Improve UI with advanced visualizations

Add real-time camera input support

## 👤 Author

Dhanushkumar Srinivasan

- Multiclass Fish Image Classification Project
