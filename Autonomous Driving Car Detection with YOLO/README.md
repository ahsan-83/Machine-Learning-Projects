# Autonomous Driving Car Detection with YOLO

The goal of this project is to implement car object detection using the YOLO (You Only Look Once) model. [Notebook](https://nbviewer.org/github/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/Autonomous_Driving_Car_Detection_with_YOLO.ipynb)<br/>
Object detection ideas were gathered from YOLO papers: [Redmon et al., 2016](https://arxiv.org/abs/1506.02640) and [Redmon and Farhadi, 2016.](https://arxiv.org/abs/1612.08242)

## YOLO Model

**You Only Look Once** (YOLO) is a powerful algorithm with high accuracy for detecting objects in real time because it requires only one forward propagation pass through the network to make predictions.
<br/>
### Inputs and outputs
- The input is a batch of images, and each image has the shape (608, 608, 3)
- The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(pc,bx,by,bh,bw,c)$

**YOLO_v2** detects 80 class objects and so the bounding box item c is 80 dimensional vector where one component is 1.

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/box_label.png" width="500"/>

### Anchor Boxes
- Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.
- The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is $(m,nH,nW,anchors,classes)$

### Encoding
- YOLO model is a Deep CNN model which converts image batch (m, 608, 608, 3) into (m, 19, 19, 5, 85) encoding.
- Grid cell on which the center of an object falls, is resposible for detecting the object.
- Each of 19x19 grid cell encodes information of 5 Anchor boxes

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/architecture.png" width="500"/>

### Bounding Box Class Scores
- Each bounding box class scores are calculated by element-wise product, scorec,i=pc×ci
- Class score with highest proabability is the detected object class type for the Bounding Box

<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/probability_extraction.png" width="500"/>

### Visualizing Bounding Box
- Calculate maximum class scores for 5 anchor boxes in each of 19x19 grid cells which represent the object class type
- Color each grid cell according to highest probability class type
- Another way to visualize is drawing bounding boxe rectangles in the image

<p float="left">
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/proba_map.png" width="450"/>
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/anchor_map.png" width="300"/>
</p>

## YOLO Bounding Box Filtering

### Bounding Box Class Score Filtering
- Filter bounding boxes based on class scores with specific threshold
- Flatten bounding box dimension from (19,19,5,85) to (19,19,425) for convenience

### Bounding Box Non-max Suppression
Non-max Suppression is a bounding box filtering technique to drop unnecessary overlapped boxes generated after class score filtering step
- Select the box with highest class score
- Compute IoU of the box with all other boxes, and remove boxes that overlap significantly (IoU >= iou_threshold)
- Repeat steps until all lower score boxes are removed


<img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/nb_images/non-max-suppression.png" width="500"/>


### YOLO Bounding Box Evaluation

- YOLO model outputs are filtered through class score filtering and non-max suppression
- YOLO model outputs are converted from box coordinates $(x,y,w,h)$ to box corners $(x1,y1,x2,y2)$
- Bounding box shapes are rescaled to plot over the orginial image as YOLO network is trained on 608 x 608 image

## YOLO Pre-trained Model

YOLO pre-trained model **YOLO_v2** based on paper [YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242) is used for Car object detection. YOLO_v2 Model is loaded from [Official YOLO website](https://pjreddie.com/darknet/yolo/) and Allan Zelener provided functions in [YAD2K](https://github.com/allanzelener/YAD2K) for converting YOLO_v2 model into Keras model. YOLO_v2 model was trained on [MS-COCO dataset](https://cocodataset.org/#home) with over 300K labeled images, 5 anchors per image and 80 object categories. <br/>

**YOLO_v2 model data can be downloaded from [MediaFire](https://www.mediafire.com/folder/j7pjxp5vpaoi1/Autonomous_Driving_Car_Detection_with_YOLO) for project.**

- In order to format the encoding of yolo_model output, yolo_head API is used provided by YAD2K in `yad2k/models/keras_yolo.py`.
- After formatting YOLO output `bounding_box_evaluation` is used to filter the bounding boxes.
- `draw_boxes` API used to draw boxes of detected object, provided by YAD2K in `yad2k/utils/utils.py`.
- `yolo_model_prediction` generates bounding box image from YOLO model from given input image

## Car Detection

### Image Car Detection

- Load image from file and detect car in the image with `yolo_model_prediction`

<p float="left">
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/images/road_image.jpg" width="300"/>
  
  <img src="https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/images/road_image_car_detection.png" width="300"/>
</p>

### Video Car Detection

- Extract frames from video file with **OpenCV** and predict car object with `yolo_video_car_detection`

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/Autonomous%20Driving%20Car%20Detection%20with%20YOLO/video/yolo_video.gif)






















