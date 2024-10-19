# Convolutional Neural Network for CIFAR-10 Classification

This project implements a deep Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, a standard dataset in computer vision tasks. The CNN is designed with various advanced techniques like **Batch Normalization**, **Residual Connections**, and **Data Augmentation** to improve the performance and generalization of the model. Additionally, the project allows testing the model on custom images and includes features like **model saving** and **loading** for training and evaluation efficiency.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Techniques Used](#techniques-used)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing the Model](#testing-the-model)
- [References](#references)

## Features

- **Convolutional Neural Network**: A multi-layer CNN designed to classify images from the CIFAR-10 dataset.
- **Batch Normalization**: Improves training stability and speeds up convergence.
- **Residual Connections**: Helps the network learn better by allowing gradients to bypass certain layers.
- **Data Augmentation**: Enhances the training dataset by applying transformations such as random cropping, flipping, and rotation to improve model generalization.
- **Custom Image Testing**: Test the model on any image by passing an image folder path.
- **Model Saving and Loading**: Automatically saves the model after training and loads it for evaluation.
- **Multi-Platform**: Supports training on devices with GPU acceleration (e.g., Apple M1 or CUDA-enabled GPUs).

## Architecture

The architecture is built with the following components:

- 5 convolutional layers with increasing depth (32, 64, 128, 256, 512 channels)
- Batch normalization after each convolutional layer
- Max pooling layers to downsample feature maps
- 2 fully connected layers to map features to CIFAR-10 classes
- Residual connections between layers to improve gradient flow

### Network Layers Overview:

- Conv2D -> BatchNorm -> ReLU -> MaxPool
- Residual Connections
- Fully Connected (FC)

## Techniques Used

1. **Batch Normalization**:
   - Helps stabilize the training process and accelerates convergence.

2. **Residual Connections**:
   - Inspired by ResNet architecture, residual blocks allow gradients to flow directly to earlier layers.

3. **Data Augmentation**:
   - Applies random transformations to the training images, increasing dataset variability and helping to prevent overfitting.

4. **Adam Optimizer**:
   - Used for optimization, offering an adaptive learning rate and faster convergence compared to stochastic gradient descent.

5. **Learning Rate Scheduling**:
   - A learning rate scheduler reduces the learning rate over time to help the model converge efficiently.

## Usage

### Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)

### Installation

1. Clone the repository

2. Install the required dependencies
```bash 
pip install -r requirements.txt
```

## Model training

To train the model, set the TRAIN_MODEL flag to True in the consts file. You can also set the SAVE_MODEL flag to store the model after training.

## Testing the model

If you already have a trained model, set the TEST_MODEL flag to True to test it on the CIFAR-10 test set.

To test the model on your own images, place your images inside the tests/ directory and run the main script. The model will make predictions on the images and output the predicted labels.

## References
	1.	CIFAR-10 Dataset: CIFAR-10
	2.	Residual Networks (ResNet): He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” ResNet Paper
	3.	Adam Optimizer: Diederik P. Kingma and Jimmy Ba, “Adam: A Method for Stochastic Optimization.” Adam Paper
	4.	PyTorch Documentation: PyTorch
	5.	Learning Rate Scheduling: Learning Rate Schedulers in PyTorch



