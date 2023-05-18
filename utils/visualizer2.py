from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QStackedWidget, QLabel, QGroupBox
from PySide6.QtCore import Signal, Slot

from PySide6.QtGui import QImage, QPixmap, QFont
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
        self.parent().parent().writeLegend()

    @Slot()
    def handleMode2Button(self):
        self.parent().setCurrentIndex(2)
        self.parent().parent().writeLegend()

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
        self.parent().parent().writeLegend()

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
        self.parent().parent().writeLegend()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.legend = {
            "0": ["press bla bla", "press bla bla", "press bla bla"],
            "1": ["press to see check box", "press bla bla", "press bla bla"],
            "2": ["press to calibrate", "press bla bla", "press bla bla"]
        }

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.mode_menu = ModeMenu()
        self.tracker_view = tracker_widgt()
        self.calib_view = calib_widjet()

        self.stackedWidget.addWidget(self.mode_menu)
        self.stackedWidget.addWidget(self.tracker_view)
        self.stackedWidget.addWidget(self.calib_view)

        legend = QGroupBox(self)
        legend.setGeometry(20, 60, 220, 150)
        legend.setTitle("Legend")
        legend.setStyleSheet("QGroupBox { font-weight: bold;}")
        self.legend_layout = QVBoxLayout()
        legend.setLayout(self.legend_layout)
        self.writeLegend()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def writeLegend(self):
        self.clearLayout(self.legend_layout)
        for l in self.legend[str(self.stackedWidget.currentIndex())]:
            label = QLabel(l)
            label.setFont(QFont("Futura", 14))
            self.legend_layout.addWidget(label)


# app = QApplication([])
#
# window = MainWindow()
# window.show()
#
# app.exec()
