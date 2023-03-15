# Machine Learning Projects

## [L-layer Deep Neural Network : Cat Image Classification](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/L-layer%20Deep%20Neural%20Network)

L-layer Deep Neural Network Model is developed from scratch with python for Cat vs non-cat image binary classification. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/L-layer%20Deep%20Neural%20Network/notebook/L-layer%20Deep%20Neural%20Network.ipynb)

**L-layer Deep Neural Network Model Architecture**

- Initialize weight and bias parameters for  L-layer deep neural network
- Compute Linear Forward Activation *LINEAR->RELU* for $L-1$ layers and *LINEAR->SIGMOID* for last layer
- Compute the loss with Binary Cross Entropy cost function
- Compute gradients of loss function respect to parameters of hidden layers in Linear Backward Activation
- Update hidden layer parameters for gradient descent using learning rate

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/model_architecture.png" width="500">

- ["Cat vs non-Cat" dataset](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/L-layer%20Deep%20Neural%20Network/datasets) stored as `train_catvnoncat.h5` and `test_catvnoncat.h5` contains 64 x 64 dimension labelled images.
- Binary Classification Model contains hidden layers with [12288, 20, 7, 5] units and output layer with 1 unit.
- Learning Rate : 0.0075
- Loss : Binary Crossentropy
- Cat vs non-cat classification model achieved 78% accuracy and 0.83 F1 Score on test dataset

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/model_loss.png" width="400"/>

## [Bird Image Classification with Transfer Learning : MobileNetV2](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/Bird%20Image%20Classification%20with%20MobileNetV2)

A transfer learning model is developed using Keras MobileNetV2 to classify 8 different species of Parrot. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/notebook/Bird_Image_Classification_with_Transfer_Learning__MobileNetV2.ipynb)

- Deep CNN model MobileNetV2 pretrained on ImageNet dataset is used for Transfer Learning.
- 900 Parrot Species images are gathered from internet with dimension 160 x 160.
- Data augmentation and normalization are applied before model training
- Keras MobileNetV2 model loaded with ImageNet weights (Freeze all layers) and top layer is replaced with Softmax classifier 
- Transfer learning model is fine-tuned by unfreezing final 30 layers of MobileNetV2 model and retrain with low learning rate
- Bird model accuracy increased from 89% to 93% after fine-tuning and loss decresed 35% to 15% in test dataset.

Model Accuracy  | Train  | Validation  | Test 
--- | --- | --- | --- 
Bird Model            | 95%    |    94%     | 89%
Bird Model Fine-tuned | 98%    |    96%     | 93%                          

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/evaluation.png" width="500"/>
<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Bird%20Image%20Classification%20with%20MobileNetV2/images/model_comparison.png" width="550"/>


## [COVID-19 Death Prediction](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/COVID-19%20Death%20Prediction)

A Deep Learning Model is developed in this project to predict death risk of COVID-19 patients. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/notebook/COVID_19_Death_Prediction.ipynb)<br/>

- [COVID-19 Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) provided by the Mexican government.
- Deep Learning Logistic Regression Model used for COVID-19 Death Prediction
- Logistic Regression Model achieved 91% test accuracy and 0.91 F1 Score

Model | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- |--- 
LR Model | 0.938971 | 0.638825 | 0.369167 | 0.467926
LR Model Under Sampled | 0.910423 | 0.882462 | 0.947661 | 0.913900 

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/model_comparison.png)

## [Autonomous Driving Car Detection with YOLO](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO)

Deep CNN model **YOLO (You Only Look Once)** is used to detect Car object in image and video. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/Autonomous_Driving_Car_Detection_with_YOLO.ipynb)<br/>

YOLO pre-trained model **YOLO_v2** based on paper [YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242) is used for Car object detection. YOLO_v2 Model is loaded from [Official YOLO website](https://pjreddie.com/darknet/yolo/) and Allan Zelener provided functions in [YAD2K](https://github.com/allanzelener/YAD2K) for converting YOLO_v2 model into Keras model. YOLO_v2 model was trained on [MS-COCO dataset](https://cocodataset.org/#home) with over 300K labeled images, 5 anchors per image and 80 object categories.


### Image Car Detection

- Load image from file and detect car in the image with `yolo_model_prediction`

<p float="left">
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/images/road_image.jpg" width="300"/>
  
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/images/road_image_car_detection.png" width="300"/>
</p>

### Video Car Detection

- Extract frames from video file with **OpenCV** and predict car object with `yolo_video_car_detection`

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/video/yolo_video.gif)











