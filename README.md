# Helmet Detection using Faster R-CNN
# Overview
This project aims to detect whether a person in an image is wearing a helmet or not, utilizing the Faster R-CNN (Region-based Convolutional Neural Network) architecture. It's particularly useful for safety monitoring in scenarios such as construction sites, bike lanes, or any environment where helmet usage is crucial.

# Features
*   **Data Preparation**: The dataset consists of images along with XML annotations indicating the presence of helmets.
*   **Model Architecture**: Utilizes the pre-trained ResNet50 model as the backbone for the Faster R-CNN architecture.
*   **Training**: The model is trained using binary cross-entropy loss and the Adam optimizer.
*   **Prediction**: After training, the model can make predictions on new images to detect helmet usage.
*   **Evaluation**: Model performance can be evaluated based on accuracy metrics.
