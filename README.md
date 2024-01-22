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

NBTRD Dataset: https://tsapps.nist.gov/NRBTD/

* Due to time limited constraint 125 train images were collected for this project and 11 test images.


### Data Labelling

For this project I used open source data annotation tool Label Studio.  https://labelstud.io/
![Screenshot 2024-01-21 225134](https://github.com/ManthanKPatel/Automatic_cartridge_masking/assets/90741568/56b70599-d105-4b9b-8590-c8129369ca86)


## 
## Environment Setup

This code is based on Mask R-CNN of https://github.com/facebookresearch/maskrcnn-benchmark.

Required Packages: python 3.9, torch 2.1 (cuda), pycocotools, opencv.

> Once the required packages are installed, run aug_data.py with reults and data folder paths as input arguments.
> When data augmentation is done and data folder is prepared run train.py fle to train the model which will save the trained model and run the model evaluations.

### Training:


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



## Install

The code is developed based on SECOND-1.5, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). Although CLOCs fusion does not need spconv, but SECOND codebase expects it to be correctly configured.

Then adding the CLOCs directory to your PYTHONPATH, you could add the following line (change '/dir/to/your/CLOCs/' according to your CLOCs directory) in your .bashrc under home directory.
```bash
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/CLOCs/'
```

## Pre-trained Models
Pretrained Models used for inference on Car, Pedestrian, and Cyclist detections can be found [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing).



Next, you could follow the SECOND-1.5 instructions to create kitti infos, reduced point cloud and groundtruth-database infos, or just download these files from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) and put them in the correct directories as shown above.

## Fusion of SECOND and Cascade-RCNN
### Preparation
CLOCs operates on the combined output of a 3D detector and a 2D detector. For this example, we use SECOND as the 3D detector, Cascade-RCNN as the 2D detector. 

1. For this example, we use detections with sigmoid scores, you could download the Cascade-RCNN detections for the KITTI train and validations set from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name:'cascade_rcnn_sigmoid_data', or you could run the 2D detector by your self and save the results for the fusion. You could also use your own 2D detector to generate these 2D detections and save them in KITTI format for fusion. 

2. Then download the pretrained SECOND models from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'second_model.zip', create an empty directory named ```model_dir``` under your CLOCs root directory and unzip the files to ```model_dir```. Your CLOCs directory should look like this:
```plain
└── CLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```

3. Then modify the config file carefully:
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/dir/to/your/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/dir/to/your/kitti_infos_train.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}
...
train_config: {
  ...
  detection_2d_path: "/dir/to/2d_detection/data"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/dir/to/your/kitti_infos_val.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}

```
### Train
```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
The trained models and related information will be saved in '/dir/to/your_model_dir'

#### Common Errors & Solutions
1.
```bash
File "./pytorch/train.py", line 869, in predict_v2
    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
RuntimeError: result type Byte can't be cast to the desired output type Bool
```
Solution: ```change opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()``` into ```opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)```. This is because SECOND-V1.5 is written in older pytorch and you have a newer version.

2. 
If you have too much NumbaWarning output messages during training/inferece that looks annoying, adding the following code at the beginning of train.py to ignore them:
```bash
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
```

### Evaluation
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/trained_model --measure_time=True --batch_size=1
```
For example if you want to test the pretrained model downloaded from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'CLOCs_SecCas_pretrained.zip', unzip it, then you could run:
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/CLOCs_SecCas_pretrained --measure_time=True --batch_size=1
```
If you want to export KITTI format label files, add ```pickle_result=False``` at the end of the above commamd.

## Pedestrian and Cyclist Detection
Step 0: **Download Pretrained Models**

If you want to utilize the already trained models, please consult the "Pre-trained Models" section above.

Step 1: **Prepare the Config Files**

Under the `./configs` directory, edit the file `pedestrian.fhd.config` or `cyclist.fhd.config`:
- Change the value of `detection_2d_path` to your pedestrian 2D detections.
- Set `steps` and `steps_per_eval` to your desired value (tweak based on training set size, for example, if we have 3712 training examples, setting “steps” equal to 74240 (3712 x 20) means training for 20 epochs. “steps_per_eval” means the step interval for each evaluation, for example, if you set “steps_per_eval” to 3712, it means doing evaluation for each epoch.)

Step 2: **Modifying `train.py`**

Edit `{CLOCs_directory}/second/pytorch/train.py`:
- Replace the paremeters in every instance of `net = build_inference_net('./configs/car.fhd.config', '../model_dir')` as it appears in the file to point towards the pedestrian/cyclist config. Additionally, ensure that the model directory points to a directory that contins the SECOND-V1.5 pretrained model for pedestrians/cyclists.
- In the `train` function, around line 258, change the `iou_bev_max` conditional in the assignment of variables `target_for_fusion`, `positive_index`, and `negative_index` to `0.5`, `0.5`, and `0.25` respectfully. The negative index value can be changed in training to alter results, but the final performance difference is not large.

Step 3: **Modifying `voxelnet.py`**

In the file `{CLOCs_directory}/scond/pytorch/models/voxelnet.py`:
- Around line 393, change `predicted_class_index = np.where(predicted_class=='Car')` to whichever detection you want, aka `'Pedestrian'` or `'Cyclist'`
- Around line 395, chnage the line `score = np.array([float(x[15]) for x in content])` to `score = np.array([float(x[15]) for x in content])/{score scale}` if your 2D detections use a score scale different that 0-1.0
- If desired, you can also change the score threshold for 2D detections around line 398. Change the `-100`, which denotes no thresholding, in the line `top_predictions = middle_predictions[np.where(middle_predictions[:,4]>=-100)]` to a decimal score threshold between 0.0 and 1.0.

Step 4: **Modify the Terminal Command**

Run the normal terminal command for training, `python ./pytorch/train.py train --config_path={config file path} --model_dir={model directory}`, pointing towards the config file we edited earlier, and the model directory with the SECOND-V1.5 pretrained model for Pedestrians/Cyclists

## Fusion of other 3D and 2D detectors
Step 1: Prepare the 2D detection candidates, run your 2D detector and save the results in KITTI format. It is recommended to run inference with NMS score threshold equals to 0 (no score thresholding), but if you don't know how to setup this, it is also fine for CLOCs.

Step 2: Prepare the 3D detection candidates, run your 3D detector and save the results in the format that SECOND could read, including a matrix with shape of N by 7 that contains the N 3D bounding boxes, and a N-element vector for the 3D confidence scores. 7 parameters correspond to the representation of a 3D bounding box. Be careful with the order and coordinate of the 7 parameters, if the parameters are in LiDAR coordinate, the order should be ```x, y, z, width, length, height, heading```; if the parameters are in camera coordinate, the orderr should be ```x, y, z, lenght, height, width, heading```. The details of the transformation functions can be found in file './second/pytorch/core/box_torch_ops.py'.

Step 3: Since the number of detection candidates are different for different 2D/3D detectors, you need to modify the corresponding parameters in the CLOCs code. Then train the CLOCs fusion. For example, there are 70400 (200x176x2) detection candidates in each frame from SECOND with batch size equals to 1. It is a very large number because SECOND is a one-stage detector, for other multi-stage detectors, you could just take the detection candidates before the final NMS function, that would reduce the number of detection candidates to hundreds or thousands.

Step 4: The output of CLOCs are fused confidence scores for all the 3D detection candidates, so you need to replace the old confidence scores (from your 3D detector) with the new fused confidence scores from CLOCs for post processing and evaluation. Then these 3D detection candidates with the corresponding CLOCs fused scores are treated as the input for your 3D detector post processing functions to generate final predictions for evaluation.
### Citation
If you find this work useful in your research, please consider citing:
```
@article{pang2020clocs,
  title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
  organization={IEEE}
}
```
### Acknowledgement
Our code are mainly based on [SECOND](https://github.com/traveller59/second.pytorch/tree/v1.5), thanks for their excellent work!
## CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection

CLOCs is a novel Camera-LiDAR Object Candidates fusion network. It provides a low-complexity multi-modal fusion framework that improves the performance of single-modality detectors. CLOCs operates on the combined output candidates of any 3D and any 2D detector, and is trained to produce more accurate 3D and 2D detection results.

### CLOCs achieves new best performance in KITTI 3D detection leaderboard (82.28%) through fusing CT3D and Cascade-RCNN, code will be updated soon.

## Environment
Tested on python3.6/3.7, pytorch 1.1.0/1.7.1, Ubuntu 16.04/18.04.

## Performance on KITTI validation set (3712 training, 3769 validation)
### CLOCs_SecCas (SECOND+Cascade-RCNN) VS SECOND:
```
new 40 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bbox: AP: 99.33 / 97.87, 93.75 / 92.37, 92.89 / 89.87
bev:  AP: 96.51 / 95.61, 92.37 / 89.54, 89.41 / 86.96
3d:   AP: 92.74 / 90.97, 82.90 / 79.94, 77.75 / 77.09
```
```
old 11 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bev:  AP: 90.52 / 90.36, 89.29 / 88.10, 87.84 / 86.80
3d:   AP: 89.49 / 88.31, 79.31 / 77.99, 77.36 / 76.52
```

### Pedestrian and Cyclist Results vs. SECOND:
Using CLOCs_SecCas (SECOND+Cascade-RCNN) trained on KITTI validation set for Pedestrians/Cyclists
```
Pedestrian:    Easy@0.5     Moderate@0.5    Hard@0.5
bbox:    AP:  74.57/58.26   70.81/54.14    62.51/50.03
bev:     AP:  68.26/61.97   62.92/56.77    56.51/51.27
3d:      AP:  62.88/58.01   56.20/51.88    50.10/47.05
```
```
Cyclist:       Easy@0.5     Moderate@0.5    Hard@0.5
bbox:    AP:  96.17/87.65   80.25/63.11    75.61/60.72
bev:     AP:  91.42/81.91   71.52/59.36    67.05/55.53
3d:      AP:  87.57/78.50   67.92/56.74    63.67/52.83
```

### To Do
 - [ ] Update the codebase to support other 3D detectors (PV-RCNN and CT3D)
 - [x] Support fusion for pedestrian and cyclists on KITTI.
 - [ ] Support easier testing for other 2D and 3D detectors, clean the code, remove unrelated environment setups to make the code easier to use.


## Install

The code is developed based on SECOND-1.5, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). Although CLOCs fusion does not need spconv, but SECOND codebase expects it to be correctly configured.

Then adding the CLOCs directory to your PYTHONPATH, you could add the following line (change '/dir/to/your/CLOCs/' according to your CLOCs directory) in your .bashrc under home directory.
```bash
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/CLOCs/'
```

## Pre-trained Models
Pretrained Models used for inference on Car, Pedestrian, and Cyclist detections can be found [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing).

## Prepare dataset (KITTI)
Download KITTI dataset and organize the files as follows:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7518 test data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── kitti_dbinfos_train.pkl
       ├── kitti_infos_train.pkl
       ├── kitti_infos_test.pkl
       ├── kitti_infos_val.pkl
       └── kitti_infos_trainval.pkl
```

Next, you could follow the SECOND-1.5 instructions to create kitti infos, reduced point cloud and groundtruth-database infos, or just download these files from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) and put them in the correct directories as shown above.

## Fusion of SECOND and Cascade-RCNN
### Preparation
CLOCs operates on the combined output of a 3D detector and a 2D detector. For this example, we use SECOND as the 3D detector, Cascade-RCNN as the 2D detector. 

1. For this example, we use detections with sigmoid scores, you could download the Cascade-RCNN detections for the KITTI train and validations set from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name:'cascade_rcnn_sigmoid_data', or you could run the 2D detector by your self and save the results for the fusion. You could also use your own 2D detector to generate these 2D detections and save them in KITTI format for fusion. 

2. Then download the pretrained SECOND models from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'second_model.zip', create an empty directory named ```model_dir``` under your CLOCs root directory and unzip the files to ```model_dir```. Your CLOCs directory should look like this:
```plain
└── CLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```

3. Then modify the config file carefully:
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/dir/to/your/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/dir/to/your/kitti_infos_train.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}
...
train_config: {
  ...
  detection_2d_path: "/dir/to/2d_detection/data"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/dir/to/your/kitti_infos_val.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}

```
### Train
```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
The trained models and related information will be saved in '/dir/to/your_model_dir'

#### Common Errors & Solutions
1.
```bash
File "./pytorch/train.py", line 869, in predict_v2
    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
RuntimeError: result type Byte can't be cast to the desired output type Bool
```
Solution: ```change opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()``` into ```opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)```. This is because SECOND-V1.5 is written in older pytorch and you have a newer version.

2. 
If you have too much NumbaWarning output messages during training/inferece that looks annoying, adding the following code at the beginning of train.py to ignore them:
```bash
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
```

### Evaluation
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/trained_model --measure_time=True --batch_size=1
```
For example if you want to test the pretrained model downloaded from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'CLOCs_SecCas_pretrained.zip', unzip it, then you could run:
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/CLOCs_SecCas_pretrained --measure_time=True --batch_size=1
```
If you want to export KITTI format label files, add ```pickle_result=False``` at the end of the above commamd.

## Pedestrian and Cyclist Detection
Step 0: **Download Pretrained Models**

If you want to utilize the already trained models, please consult the "Pre-trained Models" section above.

Step 1: **Prepare the Config Files**

Under the `./configs` directory, edit the file `pedestrian.fhd.config` or `cyclist.fhd.config`:
- Change the value of `detection_2d_path` to your pedestrian 2D detections.
- Set `steps` and `steps_per_eval` to your desired value (tweak based on training set size, for example, if we have 3712 training examples, setting “steps” equal to 74240 (3712 x 20) means training for 20 epochs. “steps_per_eval” means the step interval for each evaluation, for example, if you set “steps_per_eval” to 3712, it means doing evaluation for each epoch.)

Step 2: **Modifying `train.py`**

Edit `{CLOCs_directory}/second/pytorch/train.py`:
- Replace the paremeters in every instance of `net = build_inference_net('./configs/car.fhd.config', '../model_dir')` as it appears in the file to point towards the pedestrian/cyclist config. Additionally, ensure that the model directory points to a directory that contins the SECOND-V1.5 pretrained model for pedestrians/cyclists.
- In the `train` function, around line 258, change the `iou_bev_max` conditional in the assignment of variables `target_for_fusion`, `positive_index`, and `negative_index` to `0.5`, `0.5`, and `0.25` respectfully. The negative index value can be changed in training to alter results, but the final performance difference is not large.

Step 3: **Modifying `voxelnet.py`**

In the file `{CLOCs_directory}/scond/pytorch/models/voxelnet.py`:
- Around line 393, change `predicted_class_index = np.where(predicted_class=='Car')` to whichever detection you want, aka `'Pedestrian'` or `'Cyclist'`
- Around line 395, chnage the line `score = np.array([float(x[15]) for x in content])` to `score = np.array([float(x[15]) for x in content])/{score scale}` if your 2D detections use a score scale different that 0-1.0
- If desired, you can also change the score threshold for 2D detections around line 398. Change the `-100`, which denotes no thresholding, in the line `top_predictions = middle_predictions[np.where(middle_predictions[:,4]>=-100)]` to a decimal score threshold between 0.0 and 1.0.

Step 4: **Modify the Terminal Command**

Run the normal terminal command for training, `python ./pytorch/train.py train --config_path={config file path} --model_dir={model directory}`, pointing towards the config file we edited earlier, and the model directory with the SECOND-V1.5 pretrained model for Pedestrians/Cyclists

## Fusion of other 3D and 2D detectors
Step 1: Prepare the 2D detection candidates, run your 2D detector and save the results in KITTI format. It is recommended to run inference with NMS score threshold equals to 0 (no score thresholding), but if you don't know how to setup this, it is also fine for CLOCs.

Step 2: Prepare the 3D detection candidates, run your 3D detector and save the results in the format that SECOND could read, including a matrix with shape of N by 7 that contains the N 3D bounding boxes, and a N-element vector for the 3D confidence scores. 7 parameters correspond to the representation of a 3D bounding box. Be careful with the order and coordinate of the 7 parameters, if the parameters are in LiDAR coordinate, the order should be ```x, y, z, width, length, height, heading```; if the parameters are in camera coordinate, the orderr should be ```x, y, z, lenght, height, width, heading```. The details of the transformation functions can be found in file './second/pytorch/core/box_torch_ops.py'.

Step 3: Since the number of detection candidates are different for different 2D/3D detectors, you need to modify the corresponding parameters in the CLOCs code. Then train the CLOCs fusion. For example, there are 70400 (200x176x2) detection candidates in each frame from SECOND with batch size equals to 1. It is a very large number because SECOND is a one-stage detector, for other multi-stage detectors, you could just take the detection candidates before the final NMS function, that would reduce the number of detection candidates to hundreds or thousands.

Step 4: The output of CLOCs are fused confidence scores for all the 3D detection candidates, so you need to replace the old confidence scores (from your 3D detector) with the new fused confidence scores from CLOCs for post processing and evaluation. Then these 3D detection candidates with the corresponding CLOCs fused scores are treated as the input for your 3D detector post processing functions to generate final predictions for evaluation.
### Citation
If you find this work useful in your research, please consider citing:
```
@article{pang2020clocs,
  title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
  organization={IEEE}
}
```
### Acknowledgement
Our code are mainly based on [SECOND](https://github.com/traveller59/second.pytorch/tree/v1.5), thanks for their excellent work!

### CLOCs achieves new best performance in KITTI 3D detection leaderboard (82.28%) through fusing CT3D and Cascade-RCNN, code will be updated soon.

## Environment
Tested on python3.6/3.7, pytorch 1.1.0/1.7.1, Ubuntu 16.04/18.04.

## Performance on KITTI validation set (3712 training, 3769 validation)
### CLOCs_SecCas (SECOND+Cascade-RCNN) VS SECOND:
```
new 40 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bbox: AP: 99.33 / 97.87, 93.75 / 92.37, 92.89 / 89.87
bev:  AP: 96.51 / 95.61, 92.37 / 89.54, 89.41 / 86.96
3d:   AP: 92.74 / 90.97, 82.90 / 79.94, 77.75 / 77.09
```
```
old 11 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bev:  AP: 90.52 / 90.36, 89.29 / 88.10, 87.84 / 86.80
3d:   AP: 89.49 / 88.31, 79.31 / 77.99, 77.36 / 76.52
```

### Pedestrian and Cyclist Results vs. SECOND:
Using CLOCs_SecCas (SECOND+Cascade-RCNN) trained on KITTI validation set for Pedestrians/Cyclists
```
Pedestrian:    Easy@0.5     Moderate@0.5    Hard@0.5
bbox:    AP:  74.57/58.26   70.81/54.14    62.51/50.03
bev:     AP:  68.26/61.97   62.92/56.77    56.51/51.27
3d:      AP:  62.88/58.01   56.20/51.88    50.10/47.05
```
```
Cyclist:       Easy@0.5     Moderate@0.5    Hard@0.5
bbox:    AP:  96.17/87.65   80.25/63.11    75.61/60.72
bev:     AP:  91.42/81.91   71.52/59.36    67.05/55.53
3d:      AP:  87.57/78.50   67.92/56.74    63.67/52.83
```

### To Do
 - [ ] Update the codebase to support other 3D detectors (PV-RCNN and CT3D)
 - [x] Support fusion for pedestrian and cyclists on KITTI.
 - [ ] Support easier testing for other 2D and 3D detectors, clean the code, remove unrelated environment setups to make the code easier to use.


## Install

The code is developed based on SECOND-1.5, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). Although CLOCs fusion does not need spconv, but SECOND codebase expects it to be correctly configured.

Then adding the CLOCs directory to your PYTHONPATH, you could add the following line (change '/dir/to/your/CLOCs/' according to your CLOCs directory) in your .bashrc under home directory.
```bash
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/CLOCs/'
```

## Pre-trained Models
Pretrained Models used for inference on Car, Pedestrian, and Cyclist detections can be found [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing).

## Prepare dataset (KITTI)
Download KITTI dataset and organize the files as follows:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7518 test data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── kitti_dbinfos_train.pkl
       ├── kitti_infos_train.pkl
       ├── kitti_infos_test.pkl
       ├── kitti_infos_val.pkl
       └── kitti_infos_trainval.pkl
```

Next, you could follow the SECOND-1.5 instructions to create kitti infos, reduced point cloud and groundtruth-database infos, or just download these files from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) and put them in the correct directories as shown above.

## Fusion of SECOND and Cascade-RCNN
### Preparation
CLOCs operates on the combined output of a 3D detector and a 2D detector. For this example, we use SECOND as the 3D detector, Cascade-RCNN as the 2D detector. 

1. For this example, we use detections with sigmoid scores, you could download the Cascade-RCNN detections for the KITTI train and validations set from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name:'cascade_rcnn_sigmoid_data', or you could run the 2D detector by your self and save the results for the fusion. You could also use your own 2D detector to generate these 2D detections and save them in KITTI format for fusion. 

2. Then download the pretrained SECOND models from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'second_model.zip', create an empty directory named ```model_dir``` under your CLOCs root directory and unzip the files to ```model_dir```. Your CLOCs directory should look like this:
```plain
└── CLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```

3. Then modify the config file carefully:
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/dir/to/your/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/dir/to/your/kitti_infos_train.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}
...
train_config: {
  ...
  detection_2d_path: "/dir/to/2d_detection/data"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/dir/to/your/kitti_infos_val.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}

```
### Train
```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
The trained models and related information will be saved in '/dir/to/your_model_dir'

#### Common Errors & Solutions
1.
```bash
File "./pytorch/train.py", line 869, in predict_v2
    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
RuntimeError: result type Byte can't be cast to the desired output type Bool
```
Solution: ```change opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()``` into ```opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)```. This is because SECOND-V1.5 is written in older pytorch and you have a newer version.

2. 
If you have too much NumbaWarning output messages during training/inferece that looks annoying, adding the following code at the beginning of train.py to ignore them:
```bash
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
```

### Evaluation
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/trained_model --measure_time=True --batch_size=1
```
For example if you want to test the pretrained model downloaded from [here](https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing) file name: 'CLOCs_SecCas_pretrained.zip', unzip it, then you could run:
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/CLOCs_SecCas_pretrained --measure_time=True --batch_size=1
```
If you want to export KITTI format label files, add ```pickle_result=False``` at the end of the above commamd.

## Pedestrian and Cyclist Detection
Step 0: **Download Pretrained Models**

If you want to utilize the already trained models, please consult the "Pre-trained Models" section above.

Step 1: **Prepare the Config Files**

Under the `./configs` directory, edit the file `pedestrian.fhd.config` or `cyclist.fhd.config`:
- Change the value of `detection_2d_path` to your pedestrian 2D detections.
- Set `steps` and `steps_per_eval` to your desired value (tweak based on training set size, for example, if we have 3712 training examples, setting “steps” equal to 74240 (3712 x 20) means training for 20 epochs. “steps_per_eval” means the step interval for each evaluation, for example, if you set “steps_per_eval” to 3712, it means doing evaluation for each epoch.)

Step 2: **Modifying `train.py`**

Edit `{CLOCs_directory}/second/pytorch/train.py`:
- Replace the paremeters in every instance of `net = build_inference_net('./configs/car.fhd.config', '../model_dir')` as it appears in the file to point towards the pedestrian/cyclist config. Additionally, ensure that the model directory points to a directory that contins the SECOND-V1.5 pretrained model for pedestrians/cyclists.
- In the `train` function, around line 258, change the `iou_bev_max` conditional in the assignment of variables `target_for_fusion`, `positive_index`, and `negative_index` to `0.5`, `0.5`, and `0.25` respectfully. The negative index value can be changed in training to alter results, but the final performance difference is not large.

Step 3: **Modifying `voxelnet.py`**

In the file `{CLOCs_directory}/scond/pytorch/models/voxelnet.py`:
- Around line 393, change `predicted_class_index = np.where(predicted_class=='Car')` to whichever detection you want, aka `'Pedestrian'` or `'Cyclist'`
- Around line 395, chnage the line `score = np.array([float(x[15]) for x in content])` to `score = np.array([float(x[15]) for x in content])/{score scale}` if your 2D detections use a score scale different that 0-1.0
- If desired, you can also change the score threshold for 2D detections around line 398. Change the `-100`, which denotes no thresholding, in the line `top_predictions = middle_predictions[np.where(middle_predictions[:,4]>=-100)]` to a decimal score threshold between 0.0 and 1.0.

Step 4: **Modify the Terminal Command**

Run the normal terminal command for training, `python ./pytorch/train.py train --config_path={config file path} --model_dir={model directory}`, pointing towards the config file we edited earlier, and the model directory with the SECOND-V1.5 pretrained model for Pedestrians/Cyclists

## Fusion of other 3D and 2D detectors
Step 1: Prepare the 2D detection candidates, run your 2D detector and save the results in KITTI format. It is recommended to run inference with NMS score threshold equals to 0 (no score thresholding), but if you don't know how to setup this, it is also fine for CLOCs.

Step 2: Prepare the 3D detection candidates, run your 3D detector and save the results in the format that SECOND could read, including a matrix with shape of N by 7 that contains the N 3D bounding boxes, and a N-element vector for the 3D confidence scores. 7 parameters correspond to the representation of a 3D bounding box. Be careful with the order and coordinate of the 7 parameters, if the parameters are in LiDAR coordinate, the order should be ```x, y, z, width, length, height, heading```; if the parameters are in camera coordinate, the orderr should be ```x, y, z, lenght, height, width, heading```. The details of the transformation functions can be found in file './second/pytorch/core/box_torch_ops.py'.

Step 3: Since the number of detection candidates are different for different 2D/3D detectors, you need to modify the corresponding parameters in the CLOCs code. Then train the CLOCs fusion. For example, there are 70400 (200x176x2) detection candidates in each frame from SECOND with batch size equals to 1. It is a very large number because SECOND is a one-stage detector, for other multi-stage detectors, you could just take the detection candidates before the final NMS function, that would reduce the number of detection candidates to hundreds or thousands.

Step 4: The output of CLOCs are fused confidence scores for all the 3D detection candidates, so you need to replace the old confidence scores (from your 3D detector) with the new fused confidence scores from CLOCs for post processing and evaluation. Then these 3D detection candidates with the corresponding CLOCs fused scores are treated as the input for your 3D detector post processing functions to generate final predictions for evaluation.
### Citation
If you find this work useful in your research, please consider citing:
```
@article{pang2020clocs,
  title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
  organization={IEEE}
}
```
### Acknowledgement
Our code are mainly based on [SECOND](https://github.com/traveller59/second.pytorch/tree/v1.5), thanks for their excellent work!
