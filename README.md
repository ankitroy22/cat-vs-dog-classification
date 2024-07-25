# Cat vs Dog Classification

## Overview

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is trained on the Dogs vs Cats dataset from Kaggle and achieves a high accuracy in distinguishing between the two classes.

## Dataset

* The dataset used for this project is the Dogs vs Cats dataset from Kaggle.
* The dataset contains 25,000 images of dogs and cats, split into training and testing sets.

## Model

* The model used for this project is a CNN with the following architecture:
	+ Conv2D layer with 32 filters, kernel size 3x3, and ReLU activation
	+ Batch normalization layer
	+ Max pooling layer with pool size 2x2
	+ Conv2D layer with 64 filters, kernel size 3x3, and ReLU activation
	+ Batch normalization layer
	+ Max pooling layer with pool size 2x2
	+ Conv2D layer with 128 filters, kernel size 3x3, and ReLU activation
	+ Batch normalization layer
	+ Max pooling layer with pool size 2x2
	+ Flatten layer
	+ Dense layer with 128 units, ReLU activation, and dropout rate 0.1
	+ Dense layer with 256 units, ReLU activation, and dropout rate 0.1
	+ Dense layer with 1 unit, sigmoid activation
* The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training

* The model is trained on the training dataset for 20 epochs with a batch size of 32.
* The model's performance is evaluated on the testing dataset after each epoch.

## Results

* The model achieves a high accuracy on the testing dataset.
* The accuracy and loss curves are plotted after training.

## Usage

* To use the model, simply load a test image and preprocess it by resizing it to 256x256 pixels.
* The model's predict function can then be used to classify the image as either a cat or dog.

## Files

* `kaggle.json`: Kaggle API key file
* `dogs-vs-cats.zip`: Dogs vs Cats dataset from Kaggle
* `train` and `test` directories: Training and testing datasets
* `model.py`: Python script containing the model architecture and training code
* `README.md`: This file
