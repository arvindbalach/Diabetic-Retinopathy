# Diabetic-Retinopathy

# Code

DR.py is a python script that tries to automatically detect and grade the severity of Diabetic Retinopathy from Fundus images. I am using images from the publicly available kaggle dataset. Please click on the link to download the dataset:

https://www.kaggle.com/c/diabetic-retinopathy-detection/data

As the dataset is huge (~80 to 90 thousand images), I am trying to train a deep network on a subset of 20000 images. The images from the dataset correspond to one of the five classes. However, the distribution of data is very skewed with class four and five having very few examples. Hence to balance the data distribution, I have appropriately weighted the data in each class before training. 

Prior to training, I have also done additional preprocessing. Here are the steps:

1) As the size of the images is not the same, I resized all the images to a spatial dimension of 256x256.
2) Different images were acquired under different acquisition conditions. This has resulted in different illumination and lighting for different images. Hence, to counter these effects I have added an additional color normalization step where I divide each color image by the respective sum of individual channel images.

This is still work under progress. 

Things to do:

1) Explore new ways to reduce the influence of illumination and color.
2) Investigating different network architectures.
3) Train on all the examples.
4) I am currently loading the full training data onto a numpy array. I plan to investigate more memory efficient strategies to read the training data.

