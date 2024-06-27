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

| Defect Type   | Number | Precision (%) | Recall (%) | mAP@0.5 (%) |
|---------------|--------|---------------|------------|-------------|
| Broken        | 532    | 92.1          | 94.9       | 96.6        |
| Deformation   | 516    | 91.2          | 95.6       | 97.2        |
| Disconnection | 254    | 98.4          | 99.4       | 99.8        |
| Misalignment  | 274    | 98.5          | 99.5       | 99.6        |
| Deposition    | 407    | 99.0          | 99.8       | 99.7        |
| Obstacle      | 591    | 95.0          | 98.8       | 99.2        |
| **Total**     | 2,574  | **95.7**      | **98.0**   | **98.7**    |

