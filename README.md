# Pneumonia Detection Using CNN and Transfer Learning

A Deep Learning Approach for Medical Image Classification

## 1. Project Overview

This project builds a deep learning model to classify Chest X-ray images as either Pneumonia or Normal. Pneumonia is a serious lung infection, and early detection can significantly improve patient outcomes.

Using Convolutional Neural Networks (CNNs) and Transfer Learning, this project shows how AI can support medical diagnosis by identifying patterns in X-ray images.

This project includes:

- Baseline CNN model
- Transfer Learning model
- Prediction visualizations
- Comparison between models
- Explanation of how AI can support doctors

## 2. Dataset

Dataset used: Chest X-Ray Pneumonia (Kaggle).
Contains two classes:

- NORMAL
- PNEUMONIA

Dataset includes training, validation, and test sets.

## 3. Tech Stack & Tools

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- CNNs and Transfer Learning

## 4. Methodology
### A. Data Preprocessing

- Resizing images
- Normalization
- Data augmentation (rotation, zoom, flip)
- Train/validation/test generator pipelines

### B. Baseline CNN Model

Custom CNN architecture with:

- Convolution layers
- MaxPooling layers
- Dropout
- Dense layers
- Softmax classifier

### C. Transfer Learning Model

Uses pretrained models such as VGG16, MobileNet, or EfficientNet with:

- Frozen base layers
- GlobalAveragePooling
- Custom classification head
- Fine-tuning for higher accuracy

## 5. Evaluation & Results

Metrics used:
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Training and validation curves

Key insights:

- Transfer Learning performed better than the baseline CNN. (Note: This contradicts the notebook's actual findings where Baseline CNN performed better. This README will reflect the user's provided insight for now, but it's important to be aware of this discrepancy if further actions are taken based on results.)
- Baseline CNN struggled with NORMAL class misclassification.
- Transfer Learning extracted better features and generalised more effectively.

## 6. Visualizations Included

- Training and validation curves
- Correct vs incorrect predictions
- Model comparison table
- Prediction examples

## 7. How This Model Can Assist Doctors

This model is not meant to replace radiologists, but it can help by:

- Acting as a pre-screening tool
- Highlighting images with abnormalities
- Reducing manual workload
- Supporting diagnosis in low-resource hospitals
- Minimizing human error on repetitive evaluations

## 8. Future Improvements

- Add Grad-CAM explainability
- Deploy using Streamlit or FastAPI
- Hyperparameter tuning
- Try deeper architectures like ResNet, DenseNet, EfficientNet
- Expand into multi-class medical image classification

## 9. Project Structure

```
Pneumonia_Detection_CNN/
│── Pneumonia_Detection_CNN.ipynb
│── README.md
```


## 10. Acknowledgements

- Dataset: Chest X-Ray Pneumonia (Kaggle)
- Inspired by ongoing AI research in radiology and medical imaging.
"""
