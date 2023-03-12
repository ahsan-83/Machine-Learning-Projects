# Machine Learning Projects

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











