import sys
import os
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QCheckBox, QFileDialog
import numpy as np
import torch
import random
from PIL import Image
import datetime
import dateutil.tz

import util.visualizer_utils as utils
from options.test_options import TestOptions
import util.functions as functions
from util import keypoint_functions


def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


class Window(QMainWindow):
    def __init__(self, kp_dict, empty_image, opt):
        super().__init__()

        self.setWindowTitle(opt.title)

        self.colors = keypoint_functions.get_keypoint_colors()

        self.kp_dict = kp_dict

        random_kps = random.randint(0, len(kp_dict.keys()) - 1)
        kp_dict = kp_dict[list(kp_dict.keys())[random_kps]]
        self.current_img = list(self.kp_dict.keys())[random_kps]

        self.empty_image = empty_image

        self.opt = opt
        self.img_scale = opt.scale

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image = self.sample_image(kp_dict)
        self.UiComponents(self.image)

        self.kps = kp_dict
        self.keypoints = self.create_keypoints(kp_dict)
        self.show()

    def UiComponents(self, image):
        self.label = QLabel(self)
        pixmap = self.show_image(image)
        self.image_size = (pixmap.width(), pixmap.height())
        self.resize(pixmap.width(), pixmap.height())
        self.layout.addWidget(self.label)

        self.label_original = QLabel(self)

        self.textbox = QLabel(self)
        self.textbox.setText('Keypoint:')
        self.layout.addWidget(self.textbox)

        self.kpbutton = QPushButton('Sample New Keypoints', self)
        self.kpbutton.clicked.connect(self.new_kps)
        self.layout.addWidget(self.kpbutton)

        self.save_img = QPushButton('Save Image', self)
        self.save_img.clicked.connect(self.save_current_img)
        self.layout.addWidget(self.save_img)

    def create_keypoints(self, kp_dict, kp_size=3):
        all_kps = []
        for kp in sorted(kp_dict.keys()):
            _kp = kp_dict[kp]
            all_kps.append(utils.DragButton(self, kp, self.img_scale*(int(_kp[0]))+10, self.img_scale*(int(_kp[1]))+10,
                                            kp_size, self.opt))
        return all_kps

    def save_current_img(self):
        save_path = "sampled_imgs_{}/".format(opt.label)
        makedir(save_path)

        current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
        kp_color = (250, 30, 30)
        kp_size_in_px = 2

        self.image.save(save_path + current_time + "_generated_image.jpg")

        for kp in self.kps.keys():
            kp_x, kp_y = int(self.kps[kp][0]), int(self.kps[kp][1])
            if kp_x != -1:
                for idx1 in range(-kp_size_in_px, kp_size_in_px):
                    for idx2 in range(-kp_size_in_px, kp_size_in_px):
                        self.image.putpixel((kp_x+idx1, kp_y+idx2), kp_color)
        self.image.save(save_path + current_time + "_generated_image_w_kps.jpg")

    def get_kp_locations(self):
        kps = {}
        for kp in self.keypoints:
            kps[kp.name] = [(kp.x()-10)//self.img_scale, (kp.y()-10)//self.img_scale]
        return kps

    def show_image(self, image):
        image = image.resize((image.size[0] * self.img_scale, image.size[1] * self.img_scale))
        image = utils.convert_image(image)

        data = image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_ARGB32)

        pixmap = QtGui.QPixmap.fromImage(qim)
        self.label.setPixmap(pixmap)

        return pixmap

    def mouseReleaseEvent(self, event):
        new_kps = self.get_kp_locations()
        self.image = self.sample_image(new_kps)
        self.show_image(self.image)

    def update_keypoints(self, kp_dict):
        for idx, kp in enumerate(sorted(kp_dict.keys())):
            _kp = kp_dict[kp]
            self.keypoints[idx].move(self.img_scale*(int(_kp[0]))+10, self.img_scale*(int(_kp[1]))+10)

    def new_kps(self):
        random_kps = random.randint(0, len(self.kp_dict.keys()) - 1)
        kp_dict = self.kp_dict[list(self.kp_dict.keys())[random_kps]]
        self.current_img = list(self.kp_dict.keys())[random_kps]

        self.update_keypoints(kp_dict)
        self.image = self.sample_image(kp_dict)
        self.show_image(self.image)
        self.show()

    def sample_image(self, kps):
        self.kps = kps

        img_label = functions.generate_keypoint_condition(kps, opt)

        with torch.no_grad():
            image = netG(img_label)
            image = torch.nn.Upsample(size=[opt.image_size_y, opt.image_size_x], mode='nearest')(image)

        image = image[:, :3, :, :]
        image = torch.clamp(image, -1, 1)
        image = functions.convert_image_np(image) * 255
        image = Image.fromarray(np.uint8(image), 'RGB')

        return image


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.device = torch.device("cpu" if len(opt.gpu_ids) == 0 else "cuda:{}".format(opt.gpu_ids[0]))

    opt.name = opt.model_path
    _gpu_ids = opt.gpu_ids
    _batch_size = opt.batch_size

    opt = functions.load_config(opt)
    opt.label = opt.name

    opt.gpu_ids = _gpu_ids
    opt.name = opt.model_path
    opt.batch_size = _batch_size
    opt.title = "CharacterGAN - {}".format(opt.label)

    kp_dict = keypoint_functions.load_keypoints(opt)
    assert len(kp_dict.keys()) > 0

    netG = functions.load_model(opt)

    empty_img = np.zeros((3, opt.image_size_y, opt.image_size_x))

    app = QApplication([])
    window = Window(kp_dict, empty_img, opt)
    sys.exit(app.exec_())
