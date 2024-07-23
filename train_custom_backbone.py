# train_custom_backbone.py
from ultralytics import YOLO
from efficientnet_backbone import EfficientNetBackbone
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define custom YOLO class to integrate the custom backbone
class CustomYOLO(YOLO):
    def __init__(self, weights='yolov8s.pt', backbone=None):
        super().__init__(weights)
        if backbone:
            self.model.model[0] = backbone

if __name__ == '__main__':
    # Define custom backbone
    backbone = EfficientNetBackbone(model_name='efficientnet-b0', pretrained=True)

    # Initialize custom YOLO model with custom backbone
    model = CustomYOLO(weights='D:/yolov8/runs/detect/train126/weights/last.pt', backbone=backbone)
    dataset = 'D:/yolov8/config.yaml'
    
    # Print model summary to verify backbone replacement
    print(model.model)

    # Define training parameters
    batch_size = -1  # Use auto batch size
    initial_lr = 1e-4  # Lower learning rate for fine-tuning
    epochs = 100  # Define number of epochs

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    # Define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    results = model.train(
        data=dataset,
        resume=True,
        pretrained=True,
        save_period=1,
        batch=batch_size,
        augment=True,
        save=True,
        plots=True,
        val=True,
        imgsz=640,
        workers=8,
        project=None,
        name='train150',
        exist_ok=False,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=0.25,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        visualize=True,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        embed=None,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=None,
        workspace=4,
        nms=False,
        lr0=initial_lr,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
        cfg=None,
        tracker='botsort.yaml',
        save_dir='runs/detect/train150'
    )

    # Step the scheduler after each epoch
    scheduler.step()
