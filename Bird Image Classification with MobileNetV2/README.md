# Bird Image Classification with Transfer Learning : MobileNetV2

The project aims to classify 8 different species of Parrot using Transfer Learning. 
<br/>Deep CNN model MobileNetV2 pretrained on [ImageNet dataset](https://www.image-net.org/index.php) is used for Transfer Learning. 

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

## MobileNetV2 for Transfer Learning

MobileNetV2 was trained on ImageNet dataset and is optimized to run on mobile and other low-power applications. It's 155 layers deep and very efficient for object detection and image segmentation tasks, as well as classification tasks like this one. The architecture has three defining characteristics:

Depthwise separable convolutions
Thin input and output bottlenecks between layers
Shortcut connections between bottleneck layers
