import os
from typing import Tuple

from PySide6 import QtCore
from PySide6.QtCore import Slot, Signal, QThread
from PySide6.QtGui import QColor, QPainter, QPen, Qt, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLineEdit

class TrackerThread(QThread):
    point = Signal(int, int)

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode
        self.count = 0
    def run(self):
        while not self.isInterruptionRequested():
            point = self.model.point_of_gaze(self.mode)
            if point is None:
                continue
            x, y = point
            self.point.emit(x, y)

class ModeWidget(QWidget):
    def __init__(self, mode, model=None):
        super().__init__()

        self.mode = mode
        self.layout = QVBoxLayout()

        self.thread = TrackerThread(model, mode)
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
            painter.end()

    @Slot()
    def drawCircle(self, x, y):
        self.circle_position = (x, y)
        self.update()

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)
        self.parent().parent().writeLegend()
        self.window().showNormal()
        self.parent().parent().set_calib_data(None)

class tracker_widget(ModeWidget):
    def __init__(self, model):
        super().__init__(1, model)


class calib_widget(ModeWidget):
    def __init__(self, model):
        super().__init__(2, model)

        self.isSpacePressed = False

        self.ref_point_radius = 10  # Radius of reference points
        self.ref_point_color = QColor(255, 0, 0)  # Red color
        self.ref_points = model.ref_points if model else []

        # self.eventFilter = KeyPressFilter(parent=self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.circle_color, 3, Qt.SolidLine))

        painter.setBrush(QColor(255, 0, 0))
        for idx, point in enumerate(self.ref_points):
            if idx == self.parent().parent().model.current_ref_point:
                painter.setBrush(QColor(0, 255, 0))

            x, y = point
            painter.drawEllipse(x - self.ref_point_radius,
                                y - self.ref_point_radius,
                                self.ref_point_radius * 2,
                                self.ref_point_radius * 2)

            if painter.brush() != QColor(255, 0, 0):
                painter.setBrush(QColor(255, 0, 0))

        if self.circle_position:
            painter.setPen(QPen(self.circle_color, 3, Qt.SolidLine))
            if self.isSpacePressed:
                painter.setBrush(QColor(0, 255, 0))
                self.isSpacePressed = False
            painter.drawEllipse(self.circle_position[0] - self.circle_radius,
                                self.circle_position[1] - self.circle_radius,
                                self.circle_radius * 2,
                                self.circle_radius * 2)

        painter.end()


class FilePickerWidget(QWidget):
    def __init__(self):
        super().__init__()  # Set the mode to 2 for calibration mode

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.filenameInput.clear()
        self.filenameInput.addItems([file.name for file in os.scandir(self.calib_dir)])

    @Slot()
    def handleConfirmButton(self):

        filename = self.calib_dir + "/" + self.filenameInput.currentText()
        self.parent().parent().set_calib_data(filename)

        self.parent().setCurrentIndex(1)
        self.parent().parent().writeLegend()
        self.window().showFullScreen()

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)
        self.parent().parent().writeLegend()
        self.parent().parent().set_calib_data(None)


class FileInputWidget(QWidget):
    def __init__(self):
        super().__init__()  # Set the mode to 2 for calibration mode

        self.layout = QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignVCenter)

        self.calib_dir = "data/calib"

        self.filenameInput = QLineEdit(self)
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

        filename = self.calib_dir + "/" + self.filenameInput.text()
        self.parent().parent().set_file_data(filename)
        self.parent().parent().model.init_calib_record()

        self.parent().setCurrentIndex(2)
        self.parent().parent().writeLegend()
        self.window().showFullScreen()

    @Slot()
    def handleBackButton(self):
        self.parent().setCurrentIndex(0)
        self.parent().parent().writeLegend()
        self.parent().parent().set_calib_data(None)
