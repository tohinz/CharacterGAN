import os.path
from PIL import Image
import torch
import numpy as np
import itertools

import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from data.dataset_functions import get_params, get_transform
from util import keypoint_functions
import util.functions as functions

class KPDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.colors = keypoint_functions.get_keypoint_colors()

        images, keypoints_1d, keypoints_2d = functions.read_images_and_keypoints(opt)

        if opt.mask:
            self.images = [Image.fromarray(img, 'RGBA') for img in images]
        else:
            self.images = [Image.fromarray(img) for img in images]
        self.keypoints_1d = keypoints_1d
        self.keypoints_2d = keypoints_2d
        if opt.skeleton:
            self.skeleton = keypoint_functions.load_skeleton_info(opt)

        self.keypoints_2d = self.get_keypoints_as_rgb_image()
        self.size = (self.keypoints_2d[0].size[0], self.keypoints_2d[0].size[1])
        self.image_placeholder = np.zeros((3, self.size[0], self.size[1]))

        self.dataset_size = len(self.images)

    def get_keypoints_as_rgb_image(self):
        kp_path = os.path.join(self.opt.dir2save, "keypoints")
        functions.makedir(kp_path)

        keypoints_rgb = []
        for i, a_path in enumerate(self.keypoints_2d):
            a_path_rgb = np.zeros((3, a_path.shape[1], a_path.shape[2]))

            for idx in range(self.opt.num_keypoints):
                current_kp = np.expand_dims(a_path[idx], 0)
                current_color = np.zeros_like(a_path_rgb)
                current_color[0] = self.colors[idx][0]
                current_color[1] = self.colors[idx][1]
                current_color[2] = self.colors[idx][2]
                a_path_rgb = a_path_rgb + (np.repeat(current_kp, repeats=3, axis=0) * current_color)

            if self.opt.skeleton:
                a_path_rgb = keypoint_functions.add_skeleton(a_path_rgb, self.keypoints_1d[i], self.skeleton, self.opt)

            img = a_path_rgb * 255.
            img = img.astype(np.uint8)
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img.save(os.path.join(kp_path, "kp_{}.jpg".format(i)))
            keypoints_rgb.append(img)

        return keypoints_rgb

    def __getitem__(self, index):
        # get augmentation parameters
        params = get_params(self.opt, self.size, self.image_placeholder)

        ### get keypoint conditioning
        kps_2d = self.keypoints_2d[index]
        transform_kp = get_transform(self.opt, params, kps=True)
        kps_2d = transform_kp(kps_2d)

        ### get real image
        image = self.images[index]
        transform_img = get_transform(self.opt, params, kps=False)
        image = transform_img(image)

        input_dict = {'keypoint': kps_2d, 'image': image}
        return input_dict

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'KPDataset'

    def createInfiniteLoader(self):
        sampler = TrainingSampler(len(self))
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, self.opt.batch_size, drop_last=True)
        return torch.utils.data.DataLoader(
            dataset=self,
            num_workers=self.opt.num_workers,
            batch_sampler=batch_sampler
        )


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size, shuffle=True):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        g = torch.Generator()
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

