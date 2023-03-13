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

## Data Preprocessing

### Data Augmentation

Keras **RandomFlip** and **RandomRotation** layers are used for data augmentation in this transfer learning model.

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/data_augmentation.png" width="600"/>

### Image Normalization

Images are normalized in the range [-1,1] using `preprocess_input` function of Keras MobileNetV2 model.

## Transfer Learning Model Training

- Load Keras MobileNetV2 model with ImageNet weights (Freez all layers) and replace top layer with Softmax classifier layer for classification
- Add GlobalAveragePooling2D layer to summarize info in each channel
- Add Dropout layer to avoid overfitting
- Learning Rate : 0.001
- Optimization Algo : Adam
- Loss : Categorical Crossentropy

### Model Evaluation

Metric | Train  | Validation  | Test 
--- | --- | --- | --- |
Accuracy | 95% | 94% | 89%
Loss | 0.2317 | 0.2191 | 0.3441

### Fine-tuning Transfer Learning Model

Transfer learning model is fine-tuned by unfreezing final 30 layers of MobileNetV2 model and retrain with low learning rate.

### Fine-tuned Model Evaluation

Metric | Train  | Validation  | Test 
--- | --- | --- | --- |
Accuracy | 98% | 96% | 93%
Loss | 0.0518 | 0.0952 | 0.1620

## Model Analysis

- Loss and Accuracy both improved after fine-tuning transfer learning model

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/evaluation.png" width="500"/>

- Bird model accuracy increased from 89% to 93% after fine-tuning and loss decresed 35% to 15% in test dataset.

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/model_comparison.png" width="550"/>

















