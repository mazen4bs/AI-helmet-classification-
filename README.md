# Helmet Detection using neural network
# Overview
In this project 2 neural network models (**CNN** and **Faster RCNN**) are used to detect whether a person is wearing a helmet or not.
<br>**Note**:Each model has its own branch.
# 1-CNN model
# 2-Faster RCNN model
**Features**
*   **Data Preparation**: The dataset consists of images along with XML annotations indicating the presence of helmets.
*   **Model Architecture**: Utilizes the pre-trained ResNet50 model as the backbone for the Faster R-CNN architecture.
*   **Training**: The model is trained using binary cross-entropy loss and the Adam optimizer.
*   **Prediction**: After training, the model can make predictions on new images to detect helmet usage.
*   **Evaluation**: Model performance can be evaluated based on accuracy metrics.
