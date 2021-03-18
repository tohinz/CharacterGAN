from PyQt5 import QtCore
from PyQt5.QtWidgets import QPushButton

from PIL import Image
from . import keypoint_functions


class DragButton(QPushButton):
    def __init__(self, window, name, x_coord, y_coord, kp_size, opt):
        super().__init__(window)
        self.window = window
        self.name = name
        self.kp_labels = keypoint_functions.get_keypoint_labels(opt)
        # setting geometry of button
        self.setGeometry(x_coord, y_coord, 2*kp_size, 2*kp_size)
        # setting radius and border
        self.setStyleSheet("border-radius : {}; border: 1px solid black; background-color: black".format(kp_size))
        # adding action to a button
        self.clicked.connect(self.clickme)

    def clickme(self):
        self.window.textbox.setText("Keypoint: {}".format(self.kp_labels[self.name]))

    def mousePressEvent(self, event):
        self.__mousePressPos = None
        self.__mouseMovePos = None
        if event.button() == QtCore.Qt.LeftButton:
            self.__mousePressPos = event.globalPos()
            self.__mouseMovePos = event.globalPos()
        super(DragButton, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            # adjust offset from clicked point to origin of widget
            currPos = self.mapToGlobal(self.pos())
            globalPos = event.globalPos()
            diff = globalPos - self.__mouseMovePos
            newPos = self.mapFromGlobal(currPos + diff)
            self.move(newPos)
            self.__mouseMovePos = globalPos

        super(DragButton, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.clickme()
        if self.__mousePressPos is not None:
            moved = event.globalPos() - self.__mousePressPos
            if moved.manhattanLength() > 3:
                event.ignore()
                return
        super(DragButton, self).mouseReleaseEvent(event)


def convert_image(image):
    if image.mode == "RGB":
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
    elif image.mode == "RGBA":
        r, g, b, a = image.split()
        image = Image.merge("RGBA", (b, g, r, a))
    elif image.mode == "L":
        image = image.convert("RGBA")
    return image
