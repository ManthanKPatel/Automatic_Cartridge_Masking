# Automatic Cartridge masking with Mask R-CNN

This undertaking revolves around the automated masking of images depicting cartridge breech faces using the MaskRCNN algorithm. I have generated a labeled dataset comprising various 9 mm cartridges to distinguish five distinct regions on a cartridge, encompassing the breech face, firing pin impression, firing pin drag, and the arrow indicating drag direction.


## Mask R-CNN

I opted for Mask R-CNN for instance segmentation due to its remarkable capabilities in accurately delineating and categorizing objects within images. Mask R-CNN excels at providing precise segmentation masks for each instance in an image, making it particularly suitable for tasks such as cartridge breech face analysis. Its ability to identify and differentiate specific regions on a given object, such as the breech face, firing pin impression, and drag direction arrow in the context of this project, ensures a robust and effective solution for automatic masking. The model's versatility and performance in handling complex image segmentation tasks make it a compelling choice for achieving the desired outcomes in my project.

Mask R-CNN, or Mask Region-based Convolutional Neural Network, boasts a sophisticated architecture that combines object detection and instance segmentation seamlessly. At its core, it extends the Faster R-CNN framework by incorporating an additional branch dedicated to predicting segmentation masks for each detected object. The architecture comprises a backbone convolutional neural network (CNN) for feature extraction, a Region Proposal Network (RPN) for generating candidate object regions, and two parallel branches for bounding box regression and mask prediction. The multi-stage architecture allows for the accurate localization of objects and the precise delineation of their contours through segmentation masks. This innovative design not only facilitates object detection but also provides detailed pixel-wise segmentation, making Mask R-CNN a powerful tool for tasks requiring fine-grained analysis and identification of distinct regions within images.

<p align="center">
<img width="511" alt="Screen_Shot_2020-05-23_at_7 44 34_PM" src="https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/fba368b5-bf6d-4111-b174-f0401eadfbc5">
</p>

## Dataset

Recognizing the challenge of obtaining diverse and homogeneous cartridge images, I addressed this limitation by leveraging the NIST Ballistics Toolmark Research Database. This valuable resource provided a collection of microscopic images specifically focused on 9mm cartridges, offering a rich dataset for my project. By utilizing these images, I ensured a diverse representation of cartridge features, including the intricate details required for tasks like instance segmentation using Mask R-CNN. This strategic approach not only addressed the scarcity of relevant images but also enhanced the quality and diversity of the dataset, contributing to the robustness and effectiveness of the automatic masking solution developed in the project. 

NBTRD Dataset website: https://tsapps.nist.gov/NRBTD/

* Due to time limited constraint 125 train images were collected for this project and 11 test images.
* Data is Stored in this folder: https://drive.google.com/drive/folders/18ThzTfi_Jp5C83nGlV2-GryTzDqzr42k?usp=drive_link


### Data Labelling

For this project I used open source data annotation tool Label Studio.  https://labelstud.io/
![Screenshot 2024-01-21 225134](https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/56b70599-d105-4b9b-8590-c8129369ca86)
* Once data annotation is done, annotated images and annotations are exported as JSON file. For this project annotations were exported in COCO format.

Download dataset and organize the files as follows:

```plain
└── Segmentation
       ├── results    <-- 125 train data
       |   ├── images <-- for visualization
       |   ├── results.json
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── references     <-- 7518 test data
       |   ├── coco_evals.py 
       |   ├── coco_utils.py
       |   ├── engine.py
       |   ├── utils.py
       |   └── transforms.py 
       └── data
       ├── aug_data.py
       └── train.py
```
## Environment Setup and Training

This code is based on Mask R-CNN of https://github.com/facebookresearch/maskrcnn-benchmark.

Required Packages: python 3.9, torch 2.1 (cuda), pycocotools, opencv.

> Once the required packages are installed, run aug_data.py with reults and data folder paths as input arguments.

> When data augmentation is done and data folder is prepared run train.py fle to train the model which will save the trained model and run the model evaluations.

> Trained model is stored here: https://drive.google.com/file/d/1YmZH0hATdZgQRhjWclVV3h0A8uVqlMzx/view?usp=drive_link


### Training Evaluation:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.325
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.035
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.098
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.014
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.272
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.035
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.273
```

## Testing

To make testing process easy I have prepared a [colab notebook](https://drive.google.com/file/d/1rNj2jjm1xRfC6KFc1wh0fYplbAE5Z3wP/view?usp=sharing).

* I have imported all required libraries in colab notebook and set up model in it.
* To run this colab notebook import all the required files mentioned in it and load it to the local session.
* In colab notebook I have prepared visualisation functions to display predicted bounding box and segmentation mask data.

### Results:

#### 1. Sample test images
   
![download (4)](https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/95e9638f-216d-458f-b857-fd489d1eedfe)

#### 2. Bounding box over test images
   
![download (2)](https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/ff5d581c-4c5f-4f01-b5d3-53cfaeddfb66)

#### 3. Segmentation masks and drag arrow display over the test images
   
![download (1)](https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/c8225e33-748f-43e4-badc-e96e1bcae68d)

## References

1. [NIST Ballistics Toolmark Research Database](https://tsapps.nist.gov/NRBTD/Studies/Search)
2. [VISUALIZATION UTILITIES](https://pytorch.org/vision/0.11/auto_examples/plot_visualization_utils.html)
3. [Faster R-CNN for object detection](https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/)
4. [Automated detection of regions of interest in cartridge case images using deep learning](https://onlinelibrary.wiley.com/doi/full/10.1111/1556-4029.15319)
5. [Image matching algorithms for breech face marks and firing pins in a database of spent cartridge cases of firearms](https://www.sciencedirect.com/science/article/pii/S0379073800004205)
6. [Using GCP AutoML Vision to Predict Firearm Make/Model from Ballistic Images](https://medium.com/@dstepp2/using-gcp-automl-vision-to-predict-firearm-make-model-from-ballistic-images-55a7ca6086db)
