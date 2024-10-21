# Fish Species Classification using Artificial Neural Network (ANN)

This project focuses on classifying fish species using an Artificial Neural Network (ANN). The primary objective is to develop a model capable of distinguishing between nine different fish species using image data. The process follows a structured approach, including dataset preparation, preprocessing, model building, and evaluation. Below is a detailed breakdown of each phase.

## Table of Contents
    * Project Overview
    * Dataset Description
    * Project Workflow
      
      1. Data Loading and Exploration
      2. Data Preprocessing
      3. ANN Model Architecture
      4. Model Training
      5. Model Evaluation
      6. Results Visualization
         * Conclusion
         * Future Improvements
         * How to Run the Project

# Project Overview

This project leverages an ANN model to perform multi-class classification of fish species. The model is trained on a dataset consisting of thousands of images categorized into nine distinct fish species. The goal is to build a reliable model that can predict a fish's species based on its image, achieving high accuracy and generalization capability.

# Dataset Description
The dataset used in this project is the "A Large-Scale Fish Dataset" from Kaggle, which includes images of nine fish species:

Horse Mackerel
Gilt-Head Bream
Striped Red Mullet
Red Sea Bream
Red Mullet
Shrimp
Trout
Black Sea Sprat
Sea Bass
Each fish species has approximately 1000 images, resulting in 9000 images. The dataset is balanced, with an equal number of images per class, making it ideal for training an ANN model without bias towards any specific class.

# Project Workflow
## 1. Data Loading and Exploration

The first step involves loading the dataset and performing basic exploratory data analysis (EDA) to understand its structure. The images are organized into folders, each corresponding to a different species. During the exploration phase, we:

* Extracted File Paths: Collected the file paths for all images.
Mapped Labels: Associated each file path with the corresponding fish species label.
* Dataset Shuffle: Randomized the dataset to prevent any ordering bias during training.
* Visualizing sample images from each category gave us insights into the diversity of fish species and confirmed that the dataset was well-balanced.

## 2. Data Preprocessing

Before feeding the images into the model, several preprocessing steps were applied to ensure that the data was in the optimal format:

* Image Resizing: All images were resized to 224x224 pixels to standardize input dimensions.
* Image Augmentation: To increase the dataset’s variability and improve the model's robustness, we applied techniques like rotation, flipping, and brightness adjustment. These augmentations help the model generalize better to unseen images.
* Normalization: Pixel values were normalized to a range of [-1, 1], which helps with model convergence and improves performance.
After preprocessing, the dataset was split into training (75%), validation (10%), and test (15%) sets.

## 3. ANN Model Architecture

The classification model is built using a simple but effective Artificial Neural Network (ANN). Here’s an overview of the architecture:

* Input Layer: The input layer accepts images of size 224x224 with 3 color channels (RGB).
* Hidden Layers: The network consists of two dense (fully connected) layers, each with 128 neurons. The ReLU (Rectified Linear Unit) activation function is used for non-linearity, allowing the model to capture complex patterns in the data.
* Dropout Layers: To prevent overfitting, dropout regularization is applied after each dense layer. This randomly sets 20% of the neurons to zero during each forward pass, helping the model generalize better.
* Output Layer: The final dense layer has 9 neurons, one for each fish species, and uses the softmax activation function. This layer outputs probabilities for each class, with the class having the highest probability being selected as the prediction.

## 4. Model Training

The model is trained using the following configuration:

* Optimizer: Adam optimizer is chosen for its adaptive learning rate capabilities, which help in faster convergence.
* Loss Function: Categorical cross-entropy is used as the loss function since this is a multi-class classification problem.
* Evaluation Metric: Accuracy is monitored during training to assess model performance.
* During training, the model is evaluated on the validation set after each epoch. An early stopping callback is employed to stop training if the accuracy plateaus, preventing unnecessary computations and potential overfitting.

## 5. Model Evaluation

After training, the model is evaluated on the test set. The primary evaluation metric is accuracy, but other aspects, such as loss and confusion matrix analysis, are also considered:

Test Accuracy: The model achieved an impressive accuracy of 93.7% on the test set, indicating that it can generalize well to new images.
Confusion Matrix: A confusion matrix is used to visualize the model's performance across the different fish species, helping us identify any misclassifications or areas where the model could be improved.
## 6. Results Visualization

Several plots and visualizations were created to understand the model's performance better:

# Accuracy and Loss Curves: 

Plots showing training and validation accuracy, as well as training and validation loss over the epochs, help to assess the model’s learning process.
Sample Predictions: A set of images with their predicted labels were displayed to demonstrate how well the model performed on unseen data.
# Conclusion
This project demonstrates the successful application of an Artificial Neural Network (ANN) for image classification. Despite the simplicity of the ANN architecture, the model achieved high accuracy, proving that even simpler models can perform well in certain contexts with proper data preparation and tuning.

# Future Improvements

Several enhancements can be made to improve the model's performance and robustness:

* Deeper Architectures: Implementing more complex architectures, such as Convolutional Neural Networks (CNNs), could yield better results for image classification tasks.
* Hyperparameter Tuning: Experimenting with different hyperparameters, such as the learning rate, batch size, and number of neurons in hidden layers, could further optimize the model.
* Transfer Learning: Leveraging pre-trained models like ResNet or EfficientNet could enhance model accuracy by using learned representations from other large datasets.
* Class Imbalance: Although the dataset is balanced, experimenting with class weights or other balancing techniques may improve performance on more difficult classes.
