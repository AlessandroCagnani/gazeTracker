from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from utils.visualizer2 import MainWindow
from model import model


class controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.show()
        self.timer = QTimer()
        # Connect the timer's timeout signal to the update function
        self.timer.timeout.connect(self.update)

        # Set the timer to timeout after 1000 ms (1 second)
        self.timer.setInterval(200)
        self.timer.setSingleShot(False)
        self.timer.start()

        # self.start()

    def update(self):
        # Fetch the image from the model and update the view
        if self.view.stackedWidget.currentWidget().mode == 0:
            img = self.model.display_self()
            self.view.stackedWidget.currentWidget().update_image(img)
        else:
            points = self.model.point_of_gaze()


        # print(self.view.stackedWidget.currentWidget().mode)
    def start(self):
        #TODO: compute img and display it in the GUI
        img = self.model.display_self()
        self.view.update_image(img)





if __name__ == '__main__':
    app = QApplication([])
    controller = controller(model(), MainWindow())
    app.exec()


    print("[ MAIN ] program end")


