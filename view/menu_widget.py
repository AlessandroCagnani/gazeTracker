import cv2
import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Slot, Signal, QThread, QWaitCondition, QMutex, QMutexLocker, QEvent
from PySide6.QtGui import QColor, QPainter, QPen, Qt, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel


class ThreadMenu(QThread):
    changePixmap = Signal(QImage)

    def __init__(self, model, config, key_condition, key_mutex):
        super().__init__()
        self.model = model
        self.config = config
        self.key_condition = key_condition
        self.key_mutex = key_mutex

    def set_config(self, config):
        with QMutexLocker(self.key_mutex):
            self.config = config
            self.key_condition.wakeAll()

    def run(self):
        while not self.isInterruptionRequested():
            img = self.model.display_self(self.config)

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

        self.config = {
            "landmarks": False,
            "bbox": False,
            "head_pose": False
        }

        self.key_condition = QWaitCondition()
        self.key_mutex = QMutex()

        self.thread = ThreadMenu(model, self.config, self.key_condition, self.key_mutex)
        self.thread.changePixmap.connect(self.update_image)

        # Create a QLabel to hold the QPixmap
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.mode1Button = QPushButton("TRACK POINT OF GAZE")
        self.mode1Button.setFixedSize(200, 50)
        self.mode1Button.clicked.connect(self.handleMode1Button)
        self.layout.addWidget(self.mode1Button, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        self.mode2Button = QPushButton("CALIBRATE")
        self.mode2Button.setFixedSize(200, 50)
        self.mode2Button.clicked.connect(self.handleMode2Button)
        self.layout.addWidget(self.mode2Button, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

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
        # TODO: add the inputFileName to accept text input to save new file
        self.parent().setCurrentIndex(4)
        self.parent().parent().writeLegend()

