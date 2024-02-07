# BrTc: Brain Tumor Classification 

This project focuses on the classification of brain tumor images using convolutional neural networks (CNNs). It provides a pipeline for preprocessing the data, training different CNN models, and visualizing the results.


https://github.com/MHosseinHashemi/Brain_Tumor_Classification/assets/90381570/68a382e2-3006-44a9-a2de-69d172d10036


## Table of Contents
- Introduction
- Dependencies
- Data Preprocessing
- Model Training
- Visualizations

## Introduction
The goal of this project is to classify brain tumor images into four categories: glioma tumor, no tumor, meningioma tumor, and pituitary tumor. The code provided performs the following steps:

1. Importing necessary libraries for data processing, model training, and visualization.
2. Defining global variables such as image size and label names.
3. Creating functions for data augmentation and generators for training and testing data.
4. Building a CNN model using various pre-trained architectures like EfficientNet, VGG, MobileNet, ResNet, etc.
5. Compiling the model with an optimizer, loss function, and metrics.
6. Training the model using the training generator and validating with the validation generator.
7. Saving the best model and logging the training progress.
8. Visualizing the training history and model performance.

## Dependencies
The code requires the following libraries to be installed:
- pickle
- matplotlib
- numpy
- h5py
- tensorflow
- seaborn
- cv2
- tqdm
- os
- scikit-learn
- ipywidgets
- Pillow
- pandas
- plotly

You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
Before training the model, it is recommended to perform some preprocessing steps on the data. The code provides functions for renaming files, augmenting the training data, and resizing and shuffling the test data. You can uncomment the relevant code sections and modify the file paths according to your data location.


## Model Training
The code includes different pre-trained CNN models such as EfficientNet, VGG, MobileNet, ResNet, etc. You can choose the desired architecture by uncommenting the corresponding code block. The model is compiled with an optimizer and loss function, and various callbacks are used for monitoring and saving the best model during training. You can adjust the hyperparameters such as learning rate, batch size, and number of epochs according to your requirements.
To start training the model, uncomment the relevant code section and run the script. The model will be trained on the augmented training data and validated using the test data.


## Visualizations
After training the model, you can visualize the training history and model performance. The code provides functions for reading the training logs and plotting the validation accuracy and loss over epochs using Plotly. You can uncomment the relevant code sections and modify the file paths to visualize the results of different models.

