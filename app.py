import sys

from PySide6 import QtCore
from PySide6.QtCore import QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QGroupBox, QVBoxLayout, QLabel, QApplication

from model import model
from view.menu_widget import ModeMenu
from view.mode_widget import tracker_widget, calib_widget, FilePickerWidget, FileInputWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = model()
        self.legend = {
            "0": ["MENU", "H - display head pose",
                  "L - display landmarks", "B - display bounding box"],
            "1": ["TRACKER to see check box", "press bla bla", "press bla bla"],
            "2": ["CALIB to calibrate", "press bla bla", "press bla bla"],
            "3": ["Select file for calibration"],
            "4": ["Specify file to save"]
        }

        self.installEventFilter(self)

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.mode_menu = ModeMenu(self.model)
        self.tracker_view = tracker_widget(model=self.model)
        self.calib_view = calib_widget(model=self.model)
        self.filePicker = FilePickerWidget()
        self.inputFile = FileInputWidget()

        self.stackedWidget.addWidget(self.mode_menu)
        self.stackedWidget.addWidget(self.tracker_view)
        self.stackedWidget.addWidget(self.calib_view)
        self.stackedWidget.addWidget(self.filePicker)
        self.stackedWidget.addWidget(self.inputFile)

        legend = QGroupBox(self)
        legend.setGeometry(20, 60, 220, 150)
        legend.setTitle("Legend")
        legend.setStyleSheet("QGroupBox { font-weight: bold;}")
        self.legend_layout = QVBoxLayout()
        legend.setLayout(self.legend_layout)
        self.writeLegend()

    def eventFilter(self, obj, event):

        if event.type() == QtCore.QEvent.KeyPress:
            if self.stackedWidget.currentIndex() == 0:
                key = event.key()
                if key == QtCore.Qt.Key_B:
                    self.mode_menu.config["bbox"] = not self.mode_menu.config["bbox"]
                elif key == QtCore.Qt.Key_L:
                    self.mode_menu.config["landmarks"] = not self.mode_menu.config["landmarks"]
                elif key == QtCore.Qt.Key_H:
                    self.mode_menu.config["head_pose"] = not self.mode_menu.config["head_pose"]

                self.mode_menu.thread.set_config(self.mode_menu.config)
                # print(self.mode_menu.config)
            # elif self.stackedWidget.currentIndex() == 2:
            #     key = event.key()
            #     if key == QtCore.Qt.Key_C:
            #         self.calib_view.calibrate()
            #     elif key == QtCore.Qt.Key_N:
            #         self.calib_view.calib_next_point()
            #     elif key == QtCore.Qt.Key_S:
            #         self.calib_view.save()
            #     elif key == QtCore.Qt.Key_R:
            #         self.calib_view.reset()



        return super().eventFilter(obj, event)


    def set_calib_data(self, filename):
        self.model.set_calib_file(filename)

    def set_file_data(self, filename):
        self.model.set_file(filename)


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
