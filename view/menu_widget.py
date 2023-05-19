import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Slot, Signal, QThread
from PySide6.QtGui import QColor, QPainter, QPen, Qt, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel


class ThreadMenu(QThread):
    changePixmap = Signal(QImage)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        while not self.isInterruptionRequested():
            img = self.model.display_self()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert the img to QImage
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

            self.changePixmap.emit(qImg)

class ModeMenu(QWidget):
    def __init__(self, model):
        super().__init__()

        self.layout = QVBoxLayout()
        self.mode = 0

        self.thread = ThreadMenu(model)
        self.thread.changePixmap.connect(self.update_image)

        # Create a QLabel to hold the QPixmap
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("padding: 100px;")
        self.layout.addWidget(self.image_label)

        self.mode1Button = QPushButton("TRACK POINT OF GAZE")
        self.mode1Button.clicked.connect(self.handleMode1Button)
        self.layout.addWidget(self.mode1Button)

        self.mode2Button = QPushButton("CALIBRATE")
        self.mode2Button.clicked.connect(self.handleMode2Button)
        self.layout.addWidget(self.mode2Button)

        self.setLayout(self.layout)

    def showEvent(self, QShowEvent):
        super().showEvent(QShowEvent)
        # Start the thread when ModeMenu is shown
        if not self.thread.isRunning():
            self.thread.start()

    def hideEvent(self, QHideEvent):
        super().hideEvent(QHideEvent)
        # Stop the thread when ModeMenu is hidden
        if self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait()

    @Slot(QImage)
    def update_image(self, qimage: QImage):
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # To align the image # To align the image

    @Slot()
    def handleMode1Button(self):
        self.parent().setCurrentIndex(3)
        self.parent().parent().writeLegend()

    @Slot()
    def handleMode2Button(self):
        self.parent().setCurrentIndex(2)
        self.parent().parent().writeLegend()

