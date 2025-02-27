# 🔍 Semantic Segmentation Comparison: YOLOv8 vs YOLOv12 vs U-Net

This repository compares the performance of **YOLOv8**, **YOLOv12**, and **U-Net** for semantic segmentation tasks. The goal is to evaluate these models on a custom dataset and analyze their accuracy, speed, and resource efficiency.

## 🌟 Overview

This project evaluates three popular models for semantic segmentation:

- 🚀 **YOLOv8**: A state-of-the-art real-time object detection and segmentation model.
- 🎯 **YOLOv12**: The latest iteration of the YOLO series, optimized for segmentation tasks.
- 🏥 **U-Net**: A classic architecture widely used in medical and environmental segmentation.

The comparison focuses on:

- 📊 **Accuracy**: Mean Intersection over Union (mIoU).
- ⏱️ **Speed**: Inference time per image.
- 💾 **Resource Usage**: GPU memory consumption.

## 📥 Installation

1. Clone the repository
2. Download the dataset from [here](https://data.4tu.nl/datasets/90d13261-b0fe-444a-b408-c5a63db3d887/1).
3. Unpack the data and make sure to follow this stucture:

`data
|-- annotations
    |-- all_annotations.json
    |-- split_mapping.json
|-- images
    |-- loc1
    |-- loc2
    ....
    |-- loc6
|-- pretrained_yolo_models
    |-- different_train_sizes
    |-- generalization_loc6
    |-- trained_one_location
    |-- best_model.pt
|-- timeseries
    |-- ts_loc1_1
    |-- ts_loc1_2
    |-- ts_loc5_1`

    4. Prepare the data with`prepare_data.ipynb`, the structure should be like this:

`data
|-- annotations
    |-- split_<trainsize>_<testsize>
        |-- test
            |-- images
                |-- <imageid>.jpg
            |-- labels
                |-- <imageid>.txt
            |-- annotations.json
        |-- train
    |-- all_annotations.json
    |-- split_mapping.json
|-- combined_gt_masks
    |-- loc1.pt
    |-- loc2.pt
    ...
    |-- loc6.pt
|-- images
    |-- loc1
    |-- loc2
    ....
    |-- loc6
|-- timeseries
    |-- ts_loc1_1
    |-- ts_loc1_2
    |-- ts_loc5_1
`
