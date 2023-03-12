# Bird Image Classification with Transfer Learning : MobileNetV2

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/macaw.jpg" width="800"/>

The project aims to classify 8 different species of Parrot using Transfer Learning.<br/>
Deep CNN model **MobileNetV2** pretrained on [ImageNet dataset](https://www.image-net.org/index.php) is used for Transfer Learning. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/notebook/Bird_Image_Classification_with_Transfer_Learning__MobileNetV2.ipynb)

## Dataset

900 Parrot Species images were gathered from internet where image dimension is at least 160 x 160. Parrot Species are:

- Blue and Yellow Macaw
- Budgerigar
- Rainbow Lorikeet
- Golden Parakeet
- Hyacinth Macaw
- Spixs Macaw
- Scarlet Macaw
- Kakapo

**Dataset can be downloaded from [MediaFire](https://www.mediafire.com/file/7kdc22maou64ffw/datasets.zip/file) for this project.**

## MobileNetV2 for Transfer Learning

MobileNetV2 trained on **ImageNet dataset** is an optimized model to run on mobile which has 155 layers and very efficient for object detection.<br/>
The architecture has three defining characteristics:

- Depthwise separable convolutions
- Thin input and output bottlenecks between layers
- Shortcut connections between bottleneck layers

MobileNetV2 uses depthwise separable convolutions which is able to reduce the number of trainable parameters and operations and also speed up convolutions in two steps:

- The first step calculates an intermediate result by convolving on each of the channels independently. (Depthwise Convolution)
- Another convolution merges the outputs of the previous step into one. This gets a single result from a single feature at a time, and then is applied to all the filters in the output layer. (Pointwise Convolution)

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/mobilenetv2_architecture.png" width="600"/>

## Data Importing

- Dataset contains total 900 Train and Test set images of 8 different Parrot Species.
- Keras MobileNetV2 pretrained model accepts images of 160 x 160 or 224 x 224 dimesion. Images loaded with **160 x 160 size**.
- Train dataset is split into 8:2 ratio for training and validation data. 

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/parrot_classes.png" width="600"/>





























