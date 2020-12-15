import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QIcon, QPixmap


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.start_button = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.start_button = QPushButton("START", self)
        self.start_button.setGeometry(0,
                                      0,
                                      50,
                                      50)
        self.start_button.clicked.connect(self.handle_start_button)

        self.show()

    def handle_start_button(self):
        print('x')
        self.start_button.setGeometry(60, 60, 50, 50)
        self.start_button.setText("New")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())