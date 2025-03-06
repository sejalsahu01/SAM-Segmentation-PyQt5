import sys
import os
import torch
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QFileDialog, QListWidgetItem, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem
)
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF
from segment_anything import sam_model_registry, SamPredictor

class BoundingBox(QGraphicsRectItem):
    """A resizable and movable bounding box for object selection."""
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.setPen(QPen(QColor(255, 0, 0), 2))  # Red border
        self.setBrush(QColor(255, 0, 0, 50))  # Semi-transparent fill
        self.setFlag(QGraphicsItem.ItemIsMovable, True)  # Allow movement
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)  # Allow selection
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)  # Allow resizing

class UI_Checker(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Segmentation in PyQt5")
        self.setGeometry(100, 100, 800, 600)
        
        # Setup UI
        self.setup_ui()
        
        # Load SAM model
        self.load_sam_model()
        
        # Graphics Scene
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Bounding box properties
        self.bounding_box = None
        self.drawing = False  

    def setup_ui(self):
        """Setup UI elements."""
        layout = QtWidgets.QVBoxLayout(self)

        self.graphicsView = QtWidgets.QGraphicsView(self)
        layout.addWidget(self.graphicsView)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_segment_input = QtWidgets.QPushButton("Segment Input", self)
        self.btn_segment_input.clicked.connect(self.enable_drawing)
        btn_layout.addWidget(self.btn_segment_input)

        self.btn_analyze_segment = QtWidgets.QPushButton("Analyze Segments", self)
        self.btn_analyze_segment.clicked.connect(self.segment_with_sam)
        btn_layout.addWidget(self.btn_analyze_segment)

        self.btn_upload = QtWidgets.QPushButton("Upload Image", self)
        self.btn_upload.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.btn_upload)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Connect mouse events for manual bounding box creation
        self.graphicsView.mousePressEvent = self.start_drawing
        self.graphicsView.mouseMoveEvent = self.update_drawing
        self.graphicsView.mouseReleaseEvent = self.finish_drawing

    def load_sam_model(self):
        """Load the SAM model."""
        model_path = r"C:\Users\ASUS\Desktop\IITK\app\sam_vit_h_4b8939.pth"  # Ensure model is downloaded
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SAM model
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.predictor = SamPredictor(sam.to(device))
        self.device = device
        self.image_path = None

        print("SAM model loaded successfully!")

    def enable_drawing(self):
        """Enable manual drawing of a bounding box."""
        print("Click and drag inside graphicsView to draw a bounding box.")

    def start_drawing(self, event):
        """Start drawing a bounding box when mouse is pressed."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            scene_pos = self.graphicsView.mapToScene(event.pos())  # Convert to scene coordinates
            self.start_x = scene_pos.x()
            self.start_y = scene_pos.y()

            # Remove existing bounding box if present
            if self.bounding_box:
                self.scene.removeItem(self.bounding_box)

            # Create a new bounding box
            self.bounding_box = BoundingBox(self.start_x, self.start_y, 1, 1)
            self.scene.addItem(self.bounding_box)

    def update_drawing(self, event):
        """Resize the bounding box while dragging."""
        if self.drawing:
            scene_pos = self.graphicsView.mapToScene(event.pos())  # Convert to scene coordinates
            width = scene_pos.x() - self.start_x
            height = scene_pos.y() - self.start_y
            self.bounding_box.setRect(QRectF(self.start_x, self.start_y, width, height))

    def finish_drawing(self, event):
        """Finalize the bounding box when mouse is released."""
        if self.drawing:
            self.drawing = False
            print(f"Bounding box created at ({self.start_x}, {self.start_y}) "
                  f"with size {self.bounding_box.rect().width()}x{self.bounding_box.rect().height()}.")

    def upload_image(self):
        """Upload an image and display it in graphicsView."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            self.scene.clear()
            self.scene.addItem(QGraphicsPixmapItem(pixmap))
            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            print(f"Loaded image: {file_path}")

    def segment_with_sam(self):
        """Segment the selected region using SAM and display the mask."""
        if not self.bounding_box or not self.image_path:
            print("Error: No bounding box or image selected.")
            return

        # Load image
        image = cv2.imread(self.image_path)  # Read in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert bounding box coordinates to pixel values
        x, y, width, height = self.bounding_box.rect().getRect()
        x, y, width, height = int(x), int(y), int(width), int(height)

        # Extract the selected region
        input_image = image[y:y+height, x:x+width, :]

        # Run SAM model
        self.predictor.set_image(image)
        input_box = np.array([x, y, x + width, y + height])  # Convert to SAM input format
        masks, scores, logits = self.predictor.predict(box=input_box, multimask_output=True)

        # Get the best mask (highest confidence)
        best_mask = masks[np.argmax(scores)]

        # Display the mask in graphicsView
        self.display_mask(best_mask)

    def display_mask(self, mask):
        """Display the segmented mask in graphicsView."""
        mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8 (0-255)

        height, width = mask.shape
        image = QtGui.QImage(mask.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(image)

        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        print("Segmentation mask displayed.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UI_Checker()
    window.show()
    sys.exit(app.exec_())