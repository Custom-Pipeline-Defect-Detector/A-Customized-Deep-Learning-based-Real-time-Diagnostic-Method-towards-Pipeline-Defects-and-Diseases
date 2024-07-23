# A-Customized-Deep-Learning-based-Real-time-Diagnostic-Method-towards-Pipeline-Defects-and-Diseases
Drainage pipelines are vital for urban safety. Manual inspections are laborious and error-prone. 
This paper introduces an automated defect detection system using YOLOv8 deep learning. It achieves 98.5% precision, 98.3% recall, and 99.4% mAP@50, 
identifying six defect types and severities.
All experiments were conducted by employing Pytorch 1.7.0, Windows 11, NVIDIA GeForce RTX A2000 Ti GPU.
Descriptions of six defects from the collected drainage pipeline

  0: Deformation
  1: Obstacle
  2: Rupture
  3: Disconnect
  4: Misalignment
  5: Deposition
## YOLOv8- EfficientNet_B0 Six defect Validation Results

|  Class        | Instances  | Precision (%) | Recall (%) | mAP@50 (%) | mAP(50-95) |
|---------------|------------|---------------|------------|-------------|-----------|
| all           | 28730      | 98.5          | 98.3        | 99.4        | 97.4      | 
| Deformation   | 5676       | 97.2          | 95.8        | 99.2        | 95.5      | 
| Obstacle      | 6441       | 98.7          | 98.7        | 99.5        | 98.3      | 
| Rupture       | 5933       | 96.6          | 96.7        | 99.3        | 93.5      | 
| Disconnect    | 2804       | 99.6          | 99.2        | 99.5        | 99        | 
| Misalignment  | 3056       | 99.1          | 99.7        | 99.5        | 98.7      | 
| Deposition    | 4820       | 99.5          | 99.5        | 99.5        | 99.3      |

Modified Backbones: EfficientNet_B0
Configuration: 

| Parameter        | Value            |                    
|------------------|------------------|
| save_period      | 1                |
| batch            | `batch_size`     |
| augment          | True             |
| save             | True             |
| plots            | True             |
| val              | True             |
| imgsz            | 640              |
| workers          | 8                |
| project          | None             |
| name             | train126         |
| exist_ok         | False            |
| verbose          | True             |
| seed             | 0                |
| deterministic    | True             |
| single_cls       | False            |
| rect             | False            |
| cos_lr           | True             |
| close_mosaic     | 10               |
| amp              | True             |
| fraction         | 1.0              |
| profile          | False            |
| multi_scale      | False            |
| overlap_mask     | True             |
| mask_ratio       | 4                |
| dropout          | 0.0              |
| split            | val              |
| save_json        | False            |
| save_hybrid      | False            |
| conf             | 0.25             |
| iou              | 0.7              |
| max_det          | 300              |
| half             | False            |
| dnn              | False            |
| visualize        | True             |
| agnostic_nms     | False            |
| classes          | None             |
| retina_masks     | False            |
| embed            | None             |
| show             | False            |
| save_frames      | False            |
| save_txt         | False            |
| save_conf        | False            |
| save_crop        | False            |
| show_labels      | True             |
| show_conf        | True             |
| show_boxes       | True             |
| format           | torchscript      |
| keras            | False            |
| optimize         | False            |
| int8             | False            |
| dynamic          | False            |
| simplify         | False            |
| opset            | None             |
| workspace        | 4                |
| nms              | False            |
| lr0              | `initial_lr`     |
| lrf              | 0.01             |
| momentum         | 0.937            |
| weight_decay     | 0.0005           |
| warmup_epochs    | 3.0              |
| warmup_momentum  | 0.8              |
| warmup_bias_lr   | 0.1              |
| box              | 7.5              |
| cls              | 0.5              |
| dfl              | 1.5              |
| pose             | 12.0             |
| kobj             | 1.0              |
| label_smoothing  | 0.0              |
| nbs              | 64               |
| hsv_h            | 0.015            |
| hsv_s            | 0.7              |
| hsv_v            | 0.4              |
| degrees          | 0.0              |
| translate        | 0.1              |
| scale            | 0.5              |
| shear            | 0.0              |
| perspective      | 0.0              |
| flipud           | 0.0              |
| fliplr           | 0.5              |
| bgr              | 0.0              |
| mosaic           | 1.0              |
| mixup            | 0.0              |
| copy_paste       | 0.0              |
| auto_augment     | randaugment(ADAM)|
| erasing          | 0.4              |
| crop_fraction    | 1.0              |
| cfg              | None             |
| tracker          | botsort.yaml     |

![Untitled design](https://github.com/Custom-Pipeline-Defect-Detector/A-Customized-Deep-Learning-based-Real-time-Diagnostic-Method-towards-Pipeline-Defects-and-Diseases/assets/173538015/cfae74d8-36e0-4a7c-b373-ea8c82bfcb68)

