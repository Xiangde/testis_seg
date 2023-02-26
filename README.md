## Introduction
The repository is a 3DUNet implemented with pytorch, 
referring to this [project](https://github.com/lee-zq/3DUNet-Pytorch).
 
### Requirements:  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
### Code Structure
```angular2
├── dataset          # Training and testing dataset
│   ├── dataset_train.py 
│   ├── dataset_val.py
│   ├── dataset_test.py
│   └── transforms.py 
├── models           # Model design
│   ├── nn
│   │   └── module.py
│   │── ResUNet.py      # 3DUNet class
│   │── Unet.py      # 3DUNet class
│   │── SegNet.py      # 3DUNet class
│   └── KiUNet.py      # 3DUNet class
├── experiments           # Trained model
|── utils            # Some related tools
|   ├── common.py
|   ├── weights_init.py
|   ├── logger.py
|   ├── metrics.py
|   └── loss.py
├── preprocess.py  # preprocessing for  raw data
├── test.py          # Test code
├── train.py         # Standard training code
└── config.py        # Configuration information for training and testing
```
## Quickly Start
### 1) preprocessing: 
1. Our project dataset from Wuhan Tongji Hospital 
2. Please put the volume data and segmentation labels of trainset and testset such as:  
```
raw_dataset:
    ├── test  # 20 samples(27~46) 
    │   ├── ct
    │   │   ├── volume-27.nii
    │   │   ├── volume-28.nii
    |   |   |—— ...
    │   └── label
    │       ├── segmentation-27.nii
    │       ├── segmentation-28.nii
    |       |—— ...
    │       
    ├── train # 111 samples(0\~26 and 47\~131)
    │   ├── ct
    │   │   ├── volume-0.nii
    │   │   ├── volume-1.nii
    |   |   |—— ...
    │   └── label
    │       ├── segmentation-0.nii
    │       ├── segmentation-1.nii
    |       |—— ...
```
3. Finally, you need to change the root path of the volume data and segmentation labels in `./preprocess.py`, such as:
```
    row_dataset_path = './raw_dataset/train/'  # path of origin dataset
    fixed_dataset_path = './fixed_data/'  # path of fixed(preprocessed) dataset
```   
4. Run `python ./preprocess.py`   
If nothing goes wrong, you can see the following files in the dir `./fixed_data`
```angular2
│—— train_path_list.txt
│—— val_path_list.txt
│
|—— ct
│       volume-0.nii
│       volume-1.nii
│       volume-2.nii
│       ...
└─ label
        segmentation-0.nii
        segmentation-1.nii
        segmentation-2.nii
        ...
```  
---
### 2) Training 3DUNet
1. Firstly, you should change the some parameters in `config.py`,especially, please set `--dataset_path` to `./fixed_data`  
All parameters are commented in the file `config.py`. 
2. Secondely,run `python train.py --save model_name`  
3. Besides, you can observe the dice and loss during the training process 
in the browser through `tensorboard --logdir ./output/model_name`. 
---   
### 3) Testing 3DUNet
run `test.py`  
Please pay attention to path of trained model in `test.py`.   
(Since the calculation of the 3D convolution operation is too large,
 I use a sliding window to block the input tensor before prediction, and then stitch the results to get the final result.
 The size of the sliding window can be set by yourself in `config.py`)  

After the test, you can get the test results in the corresponding folder:`./experiments/model_name/result`
