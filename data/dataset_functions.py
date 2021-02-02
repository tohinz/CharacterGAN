from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

from util import tps_warp


def get_params(opt, size, input_im):
    w, h = size
    params = {}

    flip = random.random() > 0.5
    params["flip"] = flip

    if opt.tps_aug:
        np_im = np.array(input_im)
        tps_points_per_dim = opt.tps_points_per_dim
        src = tps_warp._get_regular_grid(np_im, points_per_dim=tps_points_per_dim)
        dst = tps_warp._generate_random_vectors(np_im, src, scale=opt.tps_scale * w)

        params["tps"] = {}
        params["tps"]["src"] = src
        params["tps"]["dst"] = dst
    return params


def get_transform(opt, params, kps):
    transform_list = []
    if opt.tps_aug:
        transform_list.append(transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if kps:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    else:
        if opt.mask:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5, 0.5))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __apply_tps(img, tps_params):
    np_im = np.array(img)
    np_im = tps_warp.tps_warp_2(np_im, tps_params['dst'], tps_params['src'])
    new_im = Image.fromarray(np_im)
    return new_im
