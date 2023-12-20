# Project Name
Dogs classification (120 types of breed)

## Table of Contents
- Dverview
- Dataset
- Creating Model
- Loading the Model into TensorFlow Lite
- Deploying the Model on AWS as a Lambda Functio

## Overview
The Dog Classification Model is an image classification project aimed at accurately identifying breeds of dogs from input images. Leveraging deep learning techniques, this project uses a pre-trained convolutional neural network (CNN) architecture to distinguish between various breeds of dogs.

## Dataset
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset 
At first I download and put to data folder after changing size. Then use train-test-split to all the paths and put each to train,val, test folders.

## Creating Model
Explain the process of creating the model. This section can include:
- Data preprocessing(At first, I downloaded the whole image files and then transport to data files by changing input size. After that divided into train, validation, test with 80%, 10%, 10% arrangement)
- Model architecture design (e.g., neural network layers)
- Training process and tuning parameters
- Best Models saving on final training
- Testing with test dataset and put accuracy. After that test with just one image 

## Loading the Model into TensorFlow Lite
Describe the steps involved in converting the model to TensorFlow Lite format. Include:
- Necessary libraries or tools required for conversion
- Conversion process from the trained model to TensorFlow Lite (using keras-image-helper) 
- Testing

## Deploying the Model on AWS as a Lambda Function
Detail the steps to deploy the TensorFlow Lite model as an AWS Lambda function. This section can include:
- Make dockerfile and then docker image
- Configuration steps for deploying TensorFlow Lite model as a Lambda function
- Set up AWS environment and Lambda function
- Testing with url



