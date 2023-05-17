# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QComboBox, QLabel
from PySide6 import QtWidgets
from PySide6.QtGui import *

class MainWindow(QMainWindow):
    def __init__(self, parent=None, mode=0):
        super().__init__(parent)
        self.setWindowTitle("Gaze Tracker")
        # setting geometry
        self.setGeometry(100, 100, 600, 400)
        self.mode = mode

        self.legend = {
            "0": ["press bla bla", "press bla bla", "press bla bla"],
            "1": ["press to see check box", "press bla bla", "press bla bla"],
            "2": ["press to calibrate", "press bla bla", "press bla bla"]
        }
        self.points = []

        self.painter = QPainter(self)
        self.initGUI()
        self.addPoint((1792 // 2, 1120 // 2), (0, 255, 0), 10)

    def initGUI(self):
        mode_selector = QComboBox(self)
        mode_selector.addItems(["Tracker", "Self Visualizer", "Calibration"])
        mode_selector.setGeometry(20, 20, 220, 30)
        mode_selector.currentIndexChanged.connect(self.mode_selector)

        legend = QtWidgets.QGroupBox(self)
        legend.setGeometry(20, 60, 220, 150)
        legend.setTitle("Legend")
        legend.setStyleSheet("QGroupBox { font-weight: bold;}")
        self.legend_layout = QtWidgets.QVBoxLayout()
        legend.setLayout(self.legend_layout)
        self.writeLegend()

    def writeLegend(self):
        self.clearLayout(self.legend_layout)
        for l in self.legend[str(self.mode)]:
            label = QtWidgets.QLabel(l)
            label.setFont(QFont("Futura", 14))
            self.legend_layout.addWidget(label)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def mode_selector(self, index):
        # TODO: connect to demo loop
        if index == 0:
            self.mode = 0
            print("Tracker")
        elif index == 1:
            self.mode = 1
            print("Self Visualizer")
        elif index == 2:
            self.mode = 2
            print("Calibration")

        self.writeLegend()

    def addPoint(self, point, color, radius):
        self.points.append((point, color, radius))
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        for point, color, radius in self.points:
            pen = QPen(QColor(*color))
            pen.setWidth(1)  # Set a small pen width
            painter.setPen(pen)
            painter.setBrush(QColor(*color))  # Set the brush to fill the ellipse
            x, y = point
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def draw_point(self, point, color, radius):
        pen = QPen(QColor(*color))
        pen.setWidth(radius * 20)
        self.painter.setPen(pen)
        x, y = point
        self.painter.drawPoint(x, y)
        self.update()



if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())
