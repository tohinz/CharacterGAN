import os
import numpy as np
import torch
from PIL import Image
import imageio
import argparse

from options.test_options import TestOptions
import util.functions as functions
from util import keypoint_functions


def draw_keypoints(kps, img, kp_color=(250, 30, 30), size_in_px=2):
    for kp in kps.keys():
        kp_x, kp_y = int(kps[kp][0]), int(kps[kp][1])
        if kp_x != -1:
            for idx1 in range(-size_in_px, size_in_px):
                for idx2 in range(-size_in_px, size_in_px):
                    img.putpixel((kp_x + idx1, kp_y + idx2), kp_color)
    return img


def generate_gif(dir2save, images, img_list, kps, draw_kps, fps):
    all_images = []
    for idx, img in enumerate(images):
        if draw_kps:
            img = draw_keypoints(kps[idx], img)
        all_images.append(img)
        img.save(os.path.join(save_path, "gen_img_{:04d}.jpg".format(idx)))

    imageio.mimsave('{}/animation_interp_{}_fps_{}.gif'.format(dir2save, len(images), fps), all_images, fps=fps)


def load_config(opt):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    model_dir = opt.name
    with open(os.path.join(model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = "-".join(param[1:])
            try:
                param_value = int(param_value)
            except:
                try:
                    param_value = float(param_value)
                except:
                    try:
                        param_value = str2bool(param_value)
                    except:
                        ValueError
            setattr(opt, param_name, param_value)
    return opt


def make_kps_in_between(img_list, kp_dict, num_in_between=10):
    all_kps = []
    for idx in range(len(img_list)-1):
        start = kp_dict[img_list[idx]]
        end = kp_dict[img_list[idx+1]]

        if idx == 0:
            all_kps.append(start)
        kps_diff = {}
        for kp in start.keys():
            x_start, y_start = int(start[kp][0]), int(start[kp][1])
            x_end, y_end = int(end[kp][0]), int(end[kp][1])
            x_diff = x_end - x_start
            y_diff = y_end - y_start
            kps_diff[kp] = [x_diff, y_diff]
            start[kp] = [x_start, y_start]
            end[kp] = [x_end, y_end]

        for i in range(1, num_in_between):
            kp_curr = {}
            for kp in kps_diff.keys():
                x_diff, y_diff = kps_diff[kp][0], kps_diff[kp][1]
                kp_curr[kp] = [int(start[kp][0] + (i/num_in_between) * x_diff), int(start[kp][1] + (i/num_in_between) * y_diff)]
            all_kps.append(kp_curr)
        all_kps.append(end)
    return all_kps


def load_image_order(opt):
    with open(opt.img_animation_list, "r") as f:
        img_animation_list = f.readlines()
    img_animation_list = [line.strip() for line in img_animation_list if line]
    img_animation_list = [img for img in img_animation_list if img != ""]
    return img_animation_list


if __name__ == '__main__':

    opt = TestOptions().parse()
    opt.device = torch.device("cpu" if len(opt.gpu_ids) == 0 else "cuda:{}".format(opt.gpu_ids[0]))

    img_animation_list = load_image_order(opt)

    opt.name = opt.model_path
    _gpu_ids = opt.gpu_ids
    _batch_size = opt.batch_size

    opt = load_config(opt)
    label = opt.name

    opt.gpu_ids = _gpu_ids
    opt.name = opt.model_path
    opt.batch_size= _batch_size
    device = "cpu" if not len(opt.gpu_ids) else "cuda:{}".format(opt.gpu_ids[0])

    kp_dict = keypoint_functions.load_keypoints(opt)
    assert len(kp_dict.keys()) > 0

    generated_images = []

    save_path = os.path.join("animations", label)
    functions.makedir(save_path)

    print("Creating keypoints...")
    img_label_between = make_kps_in_between(img_animation_list, kp_dict, opt.num_interpolations)
    img_labels = [functions.generate_keypoint_condition(kpd, opt) for kpd in img_label_between]
    layered_img_labels = []
    for layer in range(opt.num_kp_layers):
        layered_img_labels.append(torch.stack([img_labels[idx][layer] for idx in range(len(img_labels))], 0).squeeze().to(device))
    num_batches = (layered_img_labels[0].shape[0] // opt.batch_size) + 1

    print("Loading model...")
    netG = functions.load_model(opt).to(device)

    print("Generating {} images...".format(layered_img_labels[0].shape[0]))
    with torch.no_grad():
        for batch in range(num_batches):
            _img_labels = img_labels[batch * opt.batch_size : batch * opt.batch_size + opt.batch_size]
            _img_labels = []
            for idx in range(opt.num_kp_layers):
                _img_labels.append(layered_img_labels[idx][batch * opt.batch_size : batch * opt.batch_size + opt.batch_size])
            _image = netG(_img_labels)
            _image = torch.nn.Upsample(size=[opt.image_size_y, opt.image_size_x], mode='nearest')(_image).cpu()
            if batch == 0:
                image = _image
            else:
                image = torch.cat((image, _image), 0)

    images = torch.unbind(image)
    for img in images:
        img = img.unsqueeze(0)
        img = img[:, :3, :, :]
        img = torch.clamp(img, -1, 1)
        img = functions.convert_image_np(img) * 255
        img = np.uint8(img)
        img = Image.fromarray(img, 'RGB')
        generated_images.append(img)

    print("Generating animation...")
    generate_gif(save_path, generated_images, img_animation_list, img_label_between, opt.draw_kps, opt.fps)

    print("Done. Animation saved to {}".format(save_path))

