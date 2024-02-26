# Model Evaluation Metrics and Image Classification

This README demonstrates the evaluation of classification models using precision, recall, and F1-score metrics, as well as image classification using a pre-trained VGG16 model.

## Binary Classification Evaluation Metrics on MNIST Dataset

### Data Preprocessing

The script loads the MNIST dataset and preprocesses the images by normalizing pixel values and converting labels to binary format for binary classification (digits 5-9 vs. digits 0-4).

### Model Building and Training

A convolutional neural network (CNN) model is built using Keras with two convolutional layers followed by max-pooling, flattening, and dense layers. The model is trained using binary cross-entropy loss and Adam optimizer.

### Model Evaluation

The trained model is evaluated on the test set using precision, recall, and F1-score metrics.

## Multiclass Classification Evaluation Metrics on Fashion MNIST Dataset

### Data Preprocessing

The script loads the Fashion MNIST dataset and preprocesses the images by normalizing pixel values and converting labels to categorical format.

### Model Building and Training

A CNN model similar to the previous one is built using Keras for multiclass classification. The model is trained using categorical cross-entropy loss and Adam optimizer.

### Model Evaluation

The trained model is evaluated on the test set using classification report, which provides precision, recall, F1-score, and support for each class.

## Image Classification using VGG16 on Custom Images

### Image Preprocessing

The script loads custom images and preprocesses them to meet the input requirements of the VGG16 model.

### Model Loading and Prediction

The pre-trained VGG16 model is loaded with ImageNet weights. Each custom image is passed through the model for prediction.

## Dependencies

- NumPy
- TensorFlow
- Keras
