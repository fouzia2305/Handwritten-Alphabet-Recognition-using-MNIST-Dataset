# Handwritten Alphabet Recognition

This project aims to recognize handwritten alphabets (A-Z) from image data using various machine learning models. The MNIST dataset is used for training and evaluation.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Real-time Prediction](#real-time-prediction)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Problem Statement
The goal of this project is to preprocess image files from the MNIST dataset and create a pandas dataframe for classification. Subsequently, a model will be built to accurately recognize handwritten characters.

## Data Preprocessing
### Creating Numpy Arrays from Handwritten Alphabet Images
The image data is read from the extracted folders, processed, and converted into numpy arrays. This step includes:
- Reading images from folders.
- Converting images to grayscale.
- Applying thresholding.
- Flattening images.
- Storing image data and labels in numpy arrays.

## Model Building
Various machine learning models are built and trained to recognize handwritten characters. The models include:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Evaluation
The models are evaluated using metrics such as accuracy, precision, recall, and confusion matrix. Visualization of the confusion matrices helps understand the performance of each model.

## Real-time Prediction
A real-time prediction function is implemented to predict the handwritten character from a given image file. The function preprocesses the image and uses the trained SVM model to make predictions.

## Conclusion
The SVM model achieved the highest accuracy (0.9712) and demonstrated consistent performance across all metrics. Random Forest also performed well, slightly below the SVM model.

## Future Work
Future improvements could include:
- Exploring additional preprocessing techniques.
- Fine-tuning model parameters.
- Expanding the dataset for more robust training.
- Integrating real-time prediction capabilities for practical applications.

---

## Acknowledgments
This project was inspired by the classic MNIST dataset and the community of machine learning enthusiasts.
