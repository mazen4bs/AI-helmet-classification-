# Helmet Detection using neural network
# Overview
In this project 2 neural network models (**CNN** and **Faster RCNN**) are used to detect whether a person is wearing a helmet or not.
# 1-CNN model
**Loss and accuracy results**

![Screenshot 2024-05-16 163437](https://github.com/mazen4bs/Deep-learning-helmet-classification/assets/128807230/b570690d-c9ef-4b94-a7bf-c4dd8f72d586)
# 2-Faster RCNN model
**Features**
*   **Data Preparation**: The dataset consists of images along with XML annotations indicating the presence of helmets.
*   **Model Architecture**: Utilizes the pre-trained ResNet50 model as the backbone for the Faster R-CNN architecture.
*   **Training**: The model is trained using binary cross-entropy loss and the Adam optimizer.
*   **Prediction**: After training, the model can make predictions on new images to detect helmet usage.
*   **Evaluation**: Model performance can be evaluated based on accuracy metrics.
