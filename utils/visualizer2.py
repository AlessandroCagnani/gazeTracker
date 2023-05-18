from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QStackedWidget, QLabel
from PySide6.QtCore import Signal, Slot

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Signal, Qt
import cv2
import sys
import numpy as np

class ModeMenu(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.mode = 0
        # Create a QLabel to hold the QPixmap
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.mode1Button = QPushButton("TRACK POINT OF GAZE")
        self.mode1Button.clicked.connect(self.handleMode1Button)
        self.layout.addWidget(self.mode1Button)

        self.mode2Button = QPushButton("CALIBRATE")
        self.mode2Button.clicked.connect(self.handleMode2Button)
        self.layout.addWidget(self.mode2Button)

        self.setLayout(self.layout)
    def update_image(self, image: np.ndarray):
        # Convert the OpenCV image (in BGR format) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # Convert QImage to QPixmap and set it to QLabel
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # To align the image

    @Slot()
    def handleMode1Button(self):
        self.parent().setCurrentIndex(1)

    @Slot()
    def handleMode2Button(self):
        self.parent().setCurrentIndex(2)
class tracker_widgt(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.mode = 1
        self.label = QLabel("Mode 1")
        layout.addWidget(self.label)

        self.backButton = QPushButton("Back to Menu")
        self.backButton.clicked.connect(self.handleBackButton)
        layout.addWidget(self.backButton)

        self.setLayout(layout)

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)

class calib_widjet(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.mode = 2
        self.label = QLabel("Mode 2")
        layout.addWidget(self.label)

        self.backButton = QPushButton("Back to Menu")
        self.backButton.clicked.connect(self.handleBackButton)
        layout.addWidget(self.backButton)

        self.setLayout(layout)

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.stackedWidget.addWidget(ModeMenu())
        self.stackedWidget.addWidget(tracker_widgt())
        self.stackedWidget.addWidget(calib_widjet())

# app = QApplication([])
#
# window = MainWindow()
# window.show()
#
# app.exec()
