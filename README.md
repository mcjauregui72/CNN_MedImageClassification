# Medical Image Classification with Convolutional Neural Networks: Two Functional Models (Pretrained and Custom) Trained and Evaluated Individually, as an Ensemble, and Chained

## Executive Summary   
  
In this document, we trained a convolutional neural network (CNN) model from scratch to classify CT chest scan images as either cancerous or normal, and then investigated the added benefit of using a pre-trained CNN model on the same dataset by employing transfer learning and model ensembling.

The images to be classified were CT chest images from the dataset found at   https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images. Each image revealed one of the following four outcomes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, or healthy cells.

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| First Model  | 9.0000  | 0.0000  | 8.0000 | 0.0000 |
| Second Model  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Ensemble Model | 0.0000 | 0.0000 | 2.0000 | 0.0000 |
| Chained Model | 0.0000 | 0.0000 | 2.0000 | 0.0000 |

## Project Overview

The data for this project was obtained at: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images. Images reveal the presence of one of three chest carcinomas or the presence of healthy cells. The models' tasks are thus to classify each image as an one of the following four outcomes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, or healthy cells. All images were divided into training (613), testing (315), and validation (72) sets containing four types of images. 


We trained a convolutional neural network (CNN) model from scratch to classify CT chest scan images as either cancerous or normal, and then investigated the added benefit of using a pre-trained CNN model on the same dataset by employing transfer learning and model ensembling.

CNNs use convolutional and pooling layers to automatically and hierarchically learn features from images, followed by fully connected layers to classify those features into predefined categories. This process enables CNNs to effectively handle and classify complex visual data.  

Pre-training a model in the context of neural networks involves training a model on a large dataset before fine-tuning it on a specific task, with the end result known as transfer learning. Pre-training is a powerful technique, especially in scenarios where data are scarce or where training a model from scratch would be impractical due to resource constraints. In this project, we used the ResNet50 CNN, trained to classifiy 14 million images in the ImageNet dataset into 1,000 different categories, as our pre-trained model. We applied this model, by itself, and in combination with our own CNN model, to the chest images for classification. 

Because we were working with four distinct image lables classes (adenocarcinoma, large cell carcinoma, squamous cell carcinoma, and healthy cells), we chose the sparse categorical crossentropy loss function. We anticipated lables as integers because we   





