# Convolutional Neural Network for CIFAR-10 Classification

This project implements a deep Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, a standard dataset in computer vision tasks. The CNN is designed with various advanced techniques like **Batch Normalization**, **Residual Connections**, and **Data Augmentation** to improve the performance and generalization of the model. Additionally, the project allows testing the model on custom images and includes features like **model saving** and **loading** for training and evaluation efficiency.

## Table of Contents

- [Architecture](#architecture)
- [Techniques Used](#techniques-used)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing the Model](#testing-the-model)
- [Models Performance](#models-performance)
- [References](#references)
- [Summary](#summary)


## Architecture

The architecture is built with the following components:

- **Convolutional Layers**: A total of 5 convolutional layers with increasing depth (32, 64, 128, 256, 512 channels).
- **Residual Blocks**: The inclusion of multiple residual blocks enhances gradient flow and improves learning capabilities.
- **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
- **Max Pooling Layers**: Utilized to downsample feature maps, reducing spatial dimensions while retaining essential information.
- **Dropout Layers**: Incorporated after fully connected layers to prevent overfitting during training.

## Techniques Used

1. **Batch Normalization**:
   - Helps stabilize the training process and accelerates convergence.

2. **Residual Connections**:
   - Inspired by ResNet architecture, residual blocks allow gradients to flow directly to earlier layers.

3. **Data Augmentation**:
   - Applies random transformations to the training images, increasing dataset variability and helping to prevent overfitting.

4. **Dropout**:
   - A regularization technique that randomly sets a fraction of the input units to zero during training, reducing the risk of overfitting and improving model generalization.

5. **Adam Optimizer**:
   - Used for optimization, offering an adaptive learning rate and faster convergence compared to stochastic gradient descent.

6. **Learning Rate Scheduling**:
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

## Models Performance

This section outlines the performance of different models created by this project, including their accuracy metrics.

| Model Name                     | Accuracy (%) | Description                             |
|--------------------------------|--------------|-----------------------------------------|
| Cifar Net 8   | 86           | Managed to achieve this accuracy with residual blocks & number of epochs equal to 50. First model that has satisfying accuracy. |
| Cifar Net 9   | 87.40           | Managed to achieve this accuracy by increasing number of residual blocks & number of epochs to 80. |
| Cifar Net 10  |  77.77          | Increased batch size to 256 & used SGD rather than Adam. Number of epochs was decreased to 10. |
| Cifar Net 11  |     85.63       | Increased number of epochs to 50. |
| Cifar Net 12  |      87.00      | Returned to using Adam. Increased number of epochs to 80 |
| Cifar Net 16 |    87.70  | Added more residual blocks. |
| Cifar Net 18 |    87.70  | Used SGD with weight decaying & Cosine Annealing Learning Rate Scheduler |
| Cifar Net 19 |  88.80    | Unfortunately, due to bug it was deleted. Changed SGD momentum, increased number of epochs to 200. After 80 epochs there are fluctuations in improvement of accuracy. |

## Summary

Throughout this project, I successfully designed and trained multiple CNN architectures, achieving an accuracy of up to 88.80% on the CIFAR-10 classification task. By implementing advanced techniques such as residual connections and data augmentation, I enhanced the model’s performance and generalization capabilities. This experience deepened my understanding of convolutional networks and solidified my skills in utilizing PyTorch for deep learning applications.
