# A-Customized-Deep-Learning-based-Real-time-Diagnostic-Method-towards-Pipeline-Defects-and-Diseases
Drainage pipelines are vital for urban safety. Manual inspections are laborious and error-prone. 
This paper introduces an automated defect detection system using YOLOv7 deep learning. It achieves 94.6% precision, 97.3% recall, and 98.25% mAP@0.5, 
identifying six defect types and severities.
All experiments were conducted by employing Pytorch 1.7.0, Ubuntu 20.04 and an NVIDIA GeForce RTX 2080 Ti GPU.
Descriptions of six defects from the collected drainage pipeline

  0: Deformation
  1: Obstacle
  2: Broken
  3: Disconnect
  4: Misalignment
  5: Deposition
## YOLOv7-Conv2Former Six Disease Recognition Details

|  Class        | Instances  | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP50-95) |
|---------------|------------|---------------|------------|-------------|-----------|
| all           | 28730      | 98.5          | 983        | 99.4        | 97.4      | 
| Deformation   | 5676       | 97.2          | 958        | 99.2        | 95.5      | 
| Obstacle      | 6441       | 98.7          | 987        | 99.5        | 98.3      | 
| Rupture       | 5933       | 96.6          | 967        | 99.3        | 93.5      | 
| Disconnect    | 2804       | 99.6          | 992        | 99.5        | 99        | 
| Misalignment  | 3056       | 99.1          | 997        | 99.5        | 98.7      | 
| Deposition    | 4820       | 99.5          | 995        | 99.5        | 99.3      |

All experiments were conducted by employing Pytorch 1.7.0, Ubuntu 20.04 and an NVIDIA GeForce RTX 2080 Ti GPU.
![Untitled design](https://github.com/Custom-Pipeline-Defect-Detector/A-Customized-Deep-Learning-based-Real-time-Diagnostic-Method-towards-Pipeline-Defects-and-Diseases/assets/173538015/cfae74d8-36e0-4a7c-b373-ea8c82bfcb68)

