import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Load the YOLO model
model = YOLO('D:/yolov8/runs/detect/train126/weights/best.pt')

# Path to images
images_path = "D:/yolov8/data/val"

# Define class names
class_names = ['Deformation', 'Obstacle', 'Rupture', 'Disconnect', 'Misalignment', 'Deposition']

# Function to plot the results
def plot_results(image_path, predictions):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for pred in predictions:
        box = pred[:4].cpu().numpy()  # Move to CPU and convert to NumPy
        score = pred[4].cpu().item()  # Move to CPU and convert to a Python scalar
        cls = int(pred[5].cpu().item())  # Move to CPU and convert to an integer
        x1, y1, x2, y2 = box

        # Create a rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"{class_names[cls]}: {score:.2f}", color='white', fontsize=12, backgroundcolor='red')

    plt.axis('off')
    plt.show()

# Run inference and visualize results
for image_file in os.listdir(images_path):
    image_path = os.path.join(images_path, image_file)
    results = model(image_path)
    predictions = results[0].boxes.xyxy  # Extracting bounding boxes with class indices
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    # Combine predictions, confidences, and class ids
    combined_predictions = torch.cat((predictions, confidences.unsqueeze(1), class_ids.unsqueeze(1)), dim=1)

    plot_results(image_path, combined_predictions)
