import sys
import json
import cv2
import numpy as np
import winsound
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox, QScrollArea, QGridLayout, QMessageBox
from PyQt5.QtGui import QColor, QPixmap, QImage, QIcon, QFont, QPalette, QBrush
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from ultralytics import YOLO
from experta import Fact, Field, KnowledgeEngine, Rule, DefFacts, AS

# Define the Detection fact (from YOLOv8 results)
class Detection(Fact):
    name = Field(str, mandatory=True)
    confidence = Field(float, mandatory=True)
    symbolSize = Field(int, mandatory=True)
    proportion = Field(float, mandatory=True)
    severity = Field(str, mandatory=True)
    comment = Field(str, mandatory=True)  # Added comment field

# Initialize the KnowledgeBase (KB)
class ReasoningKB(KnowledgeEngine):

    def __init__(self, detections):
        super().__init__()
        self.detections = detections

    @DefFacts()
    def _initial_fact(self):
        """Load detection facts into the knowledge base."""
        for detection in self.detections:
            yield Detection(name=detection['name'], confidence=detection['confidence'],
                            symbolSize=detection['symbolSize'], proportion=detection['proportion'],
                            severity=detection['severity'], comment=detection['comment'])
            print(f"Detection added: {detection['name']} with confidence: {detection['confidence']} and symbolSize: {detection['symbolSize']}, proportion: {detection['proportion']}, severity: {detection['severity']}, comment: {detection['comment']}")

    @Rule(Detection(symbolSize=lambda symbolSize: symbolSize < 0.05))
    def low_severity(self, detection):
        detection['severity'] = 'Low'
        detection['comment'] = "Minor issue, may not require immediate action."
        print(f"Low Severity: Detected {detection['name']} with symbolSize: {detection['symbolSize']}")

    @Rule(Detection(symbolSize=lambda symbolSize: 0.05 <= symbolSize <= 0.15))
    def medium_severity(self, detection):
        detection['severity'] = 'Medium'
        detection['comment'] = "Moderate defect, requires attention but not critical."
        print(f"Medium Severity: Detected {detection['name']} with symbolSize: {detection['symbolSize']}")

    @Rule(Detection(symbolSize=lambda symbolSize: symbolSize > 0.15))
    def high_severity(self, detection):
        detection['severity'] = 'High'
        detection['comment'] = "Critical defect, immediate attention required."
        print(f"High Severity: Detected {detection['name']} with symbolSize: {detection['symbolSize']}")

    @Rule(Detection(name="Deformation"), AS.detection << Detection(proportion=lambda proportion: 0.15 < proportion <= 0.25))
    def severe_deformation(self, detection):
        detection['severity'] = 'High'
        detection['comment'] = "Critical deformation, immediate repair recommended."
        print(f"Severe Deformation detected with proportion: {detection['proportion']}")

def run_yolov8_inference(image_path, model_path):
    model = YOLO(model_path)  # Load YOLOv8 model
    results = model(image_path)  # Run inference on the image
    return results

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class PipelineDefectApp(QWidget):
    def __init__(self):
        self.available_cameras = self.get_available_cameras()  # Initialize available cameras first
        self.selected_camera = 0  # Default camera (index 0)
        self.capture = None
        super().__init__()
        self.initUI()  # Now the available_cameras attribute is initialized before calling initUI()

    def initUI(self):
        self.setWindowTitle('Pipeline Defect Detection')
        self.setWindowIcon(QIcon('icon.png'))  # Add a custom icon for the window
        self.setGeometry(100, 100, 1200, 700)

        # Set fonts and colors for buttons and background
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: 'Arial', sans-serif;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                border: 2px solid #4CAF50;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s, transform 0.2s ease;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #388e3c;
                transform: scale(1);
            }
            QTextEdit {
                background-color: #ffffff;
                border-radius: 5px;
                padding: 12px;
                font-size: 12pt;
                max-height: 600px;
                overflow-y: auto;
            }
            QLabel {
                font-size: 16px;
                color: #333333;
            }
            QComboBox {
                font-size: 14px;
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 8px;
            }
        """)

        # Create buttons and labels
        self.uploadButton = QPushButton('Upload Image', self)
        self.uploadButton.clicked.connect(self.uploadImage)

        self.detectButton = QPushButton('Detect Defects', self)
        self.detectButton.clicked.connect(self.detectDefects)

        self.captureButton = QPushButton('Start Live Camera', self)
        self.captureButton.clicked.connect(self.startCamera)

        self.cameraSelectionComboBox = QComboBox(self)
        self.cameraSelectionComboBox.addItems([f"Camera {i}" for i in range(len(self.available_cameras))])
        self.cameraSelectionComboBox.currentIndexChanged.connect(self.changeCamera)

        self.imageLabel = QLabel(self)
        self.resultText = QTextEdit(self)

        # Create a scrollable area for results
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.resultText)

        # Create warning lights (horizontal) at the bottom
        self.lightYellow = QLabel(self)
        self.lightOrange = QLabel(self)
        self.lightRed = QLabel(self)
        self.setupWarningLights()

        # Create layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.imageLabel, 3)
        h_layout.addWidget(scroll_area, 2)

        # Use a grid layout to arrange the controls neatly
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.uploadButton, 0, 0)
        grid_layout.addWidget(self.captureButton, 1, 0)
        grid_layout.addWidget(self.cameraSelectionComboBox, 2, 0)
        grid_layout.addWidget(self.detectButton, 3, 0)

        # Main vertical layout
        v_layout = QVBoxLayout()
        v_layout.addLayout(grid_layout)
        v_layout.addLayout(h_layout)

        # Add lights in a horizontal row at the bottom
        bottom_light_layout = QHBoxLayout()
        bottom_light_layout.addWidget(self.lightYellow)
        bottom_light_layout.addWidget(self.lightOrange)
        bottom_light_layout.addWidget(self.lightRed)

        # Stretch the lights to fill the width
        bottom_light_layout.setStretch(0, 1)
        bottom_light_layout.setStretch(1, 1)
        bottom_light_layout.setStretch(2, 1)

        # Add bottom layout to the main layout
        v_layout.addLayout(bottom_light_layout)

        self.setLayout(v_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        self.warning_timer = None  # Timer for repeating warning sound

    def setupWarningLights(self):
        """Set up the warning lights as horizontal rectangular shapes positioned at the bottom."""
        for light in [self.lightYellow, self.lightOrange, self.lightRed]:
            light.setFixedHeight(20)  # Keep height thinner
            light.setStyleSheet("background-color: gray; border-radius: 5px;")

    def get_available_cameras(self):
        """Detect all available cameras."""
        available_cameras = []
        for i in range(5):  # Check up to 5 devices
            capture = cv2.VideoCapture(i)
            if capture.isOpened():
                available_cameras.append(i)
                capture.release()
        return available_cameras

    def uploadImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.imagePath = file_name
            pixmap = QPixmap(file_name)
            self.imageLabel.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))  # Resize image to fit

    def changeCamera(self):
        """Change the camera based on the selected index."""
        self.selected_camera = self.cameraSelectionComboBox.currentIndex()
        if self.capture:
            self.capture.release()
        self.startCamera()

    def startCamera(self):
        """Start the live camera stream."""
        self.capture = cv2.VideoCapture(self.available_cameras[self.selected_camera])  # Use selected camera
        
        if self.capture.isOpened():
            self.captureButton.setText("Stop Live Camera")
            self.timer.start(30)  # Update frame every 30ms (about 30 FPS)
        else:
            print("Failed to open camera.")
            self.captureButton.setText("Start Live Camera")

    def process_frame(self):
        """Capture a frame from the camera and run defect detection on it."""
        if self.capture:
            ret, frame = self.capture.read()  # Read frame from webcam
            if ret:
                # Convert frame to RGB for QPixmap
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                qt_image = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.imageLabel.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))  # Resize image to fit
                
                # Save the captured frame to be used by defect detection
                self.imagePath = 'captured_frame.jpg'
                cv2.imwrite(self.imagePath, frame)
                self.detectDefects()

    def detectDefects(self):
        """Run defect detection on the uploaded or camera-captured image."""
        if hasattr(self, 'imagePath'):
            model_path = "D:/yolov8/runs/detect/train 126 backup/best.pt"
            json_path = "D:/pipeline_detection_system/static/owl_json/owl_nodes_relationships.json"

            # Load data from JSON file
            data = load_json_file(json_path)
            nodes = data.get('nodes', [])
            links = data.get('links', [])

            # Run YOLOv8 detection
            results = run_yolov8_inference(self.imagePath, model_path)
            boxes = results[0].boxes
            detections = []
            img_width, img_height = results[0].orig_shape
            high_severity_detected = False  # Flag to track high severity defects

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).flatten()
                confidence = box.conf.cpu().numpy().item()
                class_id = box.cls.cpu().numpy().item()
                name = results[0].names[class_id]
                symbolSize = int((x2 - x1) * (y2 - y1))  # Calculate symbol size as the area of the bounding box
                proportion = symbolSize / (img_width * img_height)  # Normalize by the total area of the image

                # Normalize symbol size to be between 0 and 1
                normalized_symbol_size = symbolSize / (img_width * img_height)

                # Initialize severity based on normalized symbol size
                if normalized_symbol_size < 0.05:
                    severity = 'Low'
                    comment = "Minor issue, may not require immediate action."
                    self.updateLights("yellow")
                elif 0.05 <= normalized_symbol_size <= 0.15:
                    severity = 'Medium'
                    comment = "Moderate defect, requires attention but not critical."
                    self.updateLights("orange")
                else:
                    severity = 'High'
                    comment = "Critical defect, immediate attention required."
                    high_severity_detected = True  # Set the flag for high severity
                    self.updateLights("red")
                    self.playWarningSound()  # Play warning sound in loop and pop-up

                detections.append({
                    'name': name,
                    'confidence': confidence,
                    'symbolSize': symbolSize,
                    'proportion': proportion,
                    'severity': severity,
                    'comment': comment,
                    'box': [x1, y1, x2, y2]  # Store bounding box coordinates
                })

            # Draw the bounding boxes on the image
            self.drawBoundingBoxes(detections)

            # Initialize and run the reasoning engine with severity calculation
            kb = ReasoningKB(detections)
            kb.reset()
            kb.run()

            # Display results
            results_text = "Detected Defects:\n\n"
            for detection in detections:
                results_text += f"Detected: {detection['name']}\n"
                results_text += f"Confidence: {detection['confidence']:.2f}\n"
                results_text += f"Symbol Size: {detection['symbolSize']}\n"
                results_text += f"Proportion: {detection['proportion']:.4f}\n"
                results_text += f"Severity: {detection['severity']}\n"
                results_text += f"Comment: {detection['comment']}\n\n"

            self.resultText.setText(results_text)

    def updateLights(self, severity):
        """Update the warning lights based on severity."""
        if severity == "red":
            self.lightRed.setStyleSheet("background-color: red;")
            self.lightYellow.setStyleSheet("background-color: yellow;")
            self.lightOrange.setStyleSheet("background-color: orange;")
        elif severity == "orange":
            self.lightRed.setStyleSheet("background-color: gray;")
            self.lightYellow.setStyleSheet("background-color: yellow;")
            self.lightOrange.setStyleSheet("background-color: orange;")
        elif severity == "yellow":
            self.lightRed.setStyleSheet("background-color: gray;")
            self.lightYellow.setStyleSheet("background-color: yellow;")
            self.lightOrange.setStyleSheet("background-color: gray;")

    def playWarningSound(self):
        """Play a warning sound in a loop until OK is pressed."""
        self.warning_timer = QTimer(self)
        self.warning_timer.timeout.connect(self._playWarningSound)
        self.warning_timer.start(1000)  # Repeat every second

        # Display a warning message
        reply = QMessageBox.warning(self, "Attention Required", "A high severity defect has been detected. Immediate attention required!", QMessageBox.Ok)
        
        if reply == QMessageBox.Ok:
            self.warning_timer.stop()  # Stop the sound when OK is pressed

    def _playWarningSound(self):
        """Play warning sound every second."""
        winsound.Beep(1000, 500)  # Frequency, duration (in milliseconds)

    def drawBoundingBoxes(self, detections):
        """Draw bounding boxes on the image"""
        image = cv2.imread(self.imagePath)
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw rectangle for bounding box
            label = detection['name']
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1, y1 - 10), font, 0.9, (255, 255, 255), 2)  # Draw label

        cv2.imwrite(self.imagePath, image)
        pixmap = QPixmap(self.imagePath)
        self.imageLabel.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PipelineDefectApp()
    ex.show()
    sys.exit(app.exec_())
