import sys

from PySide6.QtCore import QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QGroupBox, QVBoxLayout, QLabel, QApplication

from model import model
from view.menu_widget import ModeMenu
from view.mode_widget import tracker_widget, calib_widget, FileInputWidget



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = model()
        self.legend = {
            "0": ["press bla bla", "press bla bla", "press bla bla"],
            "1": ["press to see check box", "press bla bla", "press bla bla"],
            "2": ["press to calibrate", "press bla bla", "press bla bla"],
            "3": ["Select file for calibration"],
            "4": ["Specify file to save"]
        }

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.mode_menu = ModeMenu(self.model)
        self.tracker_view = tracker_widget(model=self.model)
        self.calib_view = calib_widget(model=self.model)
        self.filePicker = FileInputWidget()

        self.stackedWidget.addWidget(self.mode_menu)
        self.stackedWidget.addWidget(self.tracker_view)
        self.stackedWidget.addWidget(self.calib_view)
        self.stackedWidget.addWidget(self.filePicker)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
