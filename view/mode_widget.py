import os
from typing import Tuple

from PySide6 import QtCore
from PySide6.QtCore import Slot, Signal, QThread
from PySide6.QtGui import QColor, QPainter, QPen, Qt, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox


class TrackerThread(QThread):
    point = Signal(int, int)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        while not self.isInterruptionRequested():
            x, y = self.model.point_of_gaze()
            self.point.emit(x, y)


class ModeWidget(QWidget):
    def __init__(self, mode, model=None):
        super().__init__()

        self.mode = mode
        self.layout = QVBoxLayout()

        self.thread = TrackerThread(model)
        self.thread.point.connect(self.drawCircle)

        self.backButton = QPushButton("Back to Menu")
        self.backButton.setFixedSize(200, 50)
        self.backButton.clicked.connect(self.handleBackButton)
        self.layout.addWidget(self.backButton, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        self.circle_radius = 10
        self.circle_color = QColor(255, 0, 0)  # Red color
        self.circle_position = None

    def showEvent(self, QShowEvent):
        super().showEvent(QShowEvent)

        if not self.thread.isRunning():
            self.thread.start()

    def hideEvent(self, QHideEvent):
        super().hideEvent(QHideEvent)
        if self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait()


    def paintEvent(self, event):
        if self.circle_position:
            painter = QPainter(self)
            painter.setPen(QPen(self.circle_color, 3, Qt.SolidLine))
            painter.drawEllipse(self.circle_position[0] - self.circle_radius,
                                self.circle_position[1] - self.circle_radius,
                                self.circle_radius * 2,
                                self.circle_radius * 2)

    @Slot()
    def drawCircle(self, x, y):
        self.circle_position = (x, y)
        self.update()

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)
        self.parent().parent().writeLegend()
        self.window().showNormal()

class tracker_widget(ModeWidget):
    def __init__(self, model):
        super().__init__(1, model)


class calib_widget(ModeWidget):
    def __init__(self, model):
        super().__init__(2, model)


class FileInputWidget(QWidget):
    def __init__(self):
        super().__init__()  # Set the mode to 2 for calibration mode

        self.mode = 3
        self.layout = QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignVCenter)

        self.calib_dir = "data/calib"

        self.filenameInput = QComboBox(self)
        self.filenameInput.addItems([file.name for file in os.scandir(self.calib_dir)])
        self.filenameInput.setFixedSize(250, 20)
        self.layout.addWidget(self.filenameInput, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)


        self.confimButton = QPushButton("CONFIRM")
        self.confimButton.setFixedSize(200, 50)
        self.confimButton.clicked.connect(self.handleConfirmButton)
        self.layout.addWidget(self.confimButton, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        self.backButton = QPushButton("Back to Menu")
        self.backButton.setFixedSize(200, 50)
        self.backButton.clicked.connect(self.handleBackButton)
        self.layout.addWidget(self.backButton, alignment=QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

    @Slot()
    def handleConfirmButton(self):
        # Retrieve the filename before switching back to the menu
        self.parent().setCurrentIndex(1)
        self.parent().parent().writeLegend()
        self.window().showFullScreen()

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)
        self.parent().parent().writeLegend()