import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as img
from skimage import color, filters, morphology
import os
import glob
from PIL import Image
import torchvision.transforms as transforms

from . import keypoint_functions


def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def denorm(x):
    if torch.min(x) < -1 or torch.max(x) > 1:
        return _normalize(x)
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def _normalize(tensor):
        tensor = tensor.clone()  # avoid modifying tensor in-place

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            return img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t):
            return norm_ip(t, float(t.min()), float(t.max()))

        tensor = norm_range(tensor)
        return tensor


def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = inp[-1,:,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = inp[-1,-1,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp


def save_image(name, image):
    plt.imsave(name, convert_image_np(image), vmin=0, vmax=1)


def read_images_and_keypoints(opt):
    imgs = glob.glob(os.path.join(opt.dataroot, "*.jpg")) + glob.glob(os.path.join(opt.dataroot, "*.png")) + glob.glob(os.path.join(opt.dataroot, "*.jpeg"))
    keypoints = keypoint_functions.load_keypoints(opt)

    images = []
    keypoints_1d = []
    keypoints_2d = []
    num_kps = opt.num_keypoints

    # load images and corresponding keypoints
    for _img in sorted(imgs):
        name = _img.split("/")[-1].split(".")[0]
        x = img.imread(_img)
        x = x[:, :, :3]

        # automatically construct the mask based on background color
        if opt.mask:
            save_dir = os.path.join(opt.dir2save, "masks")
            makedir(save_dir)
            alpha = np.ones_like(x[:, :, 0])
            alpha[np.isclose(np.mean(x, axis=2), opt.bkg_color, rtol=1e-1)] = 0

            alpha = np.array(alpha, dtype=bool)
            alpha = morphology.remove_small_objects(alpha, 10, connectivity=1)
            alpha = morphology.remove_small_holes(alpha, 2, connectivity=2)
            alpha = np.array(alpha, dtype=float)

            alpha = np.expand_dims(alpha, -1)
            alpha_img = np.repeat(alpha, 3, axis=2)
            alpha = alpha * 255
            plt.imsave(os.path.join(save_dir, "mask_{}.jpg".format(name)), alpha_img, vmin=0, vmax=255)
            alpha = alpha.astype(np.uint8)

        # load corresponding keypoints for current image
        try:
            img_keypoints = keypoints[_img.split("/")[-1]]
        except KeyError:
            print("Found no matching keypoints for {}...skipping this image.".format(name))
            continue

        # normalize keypoint conditioning
        x_condition = keypoint_functions.create_keypoint_condition(x, img_keypoints, opt, num_kps)
        x_condition = (x_condition + 1) / 2.0

        if opt.mask:
            x = np.concatenate([x, alpha], -1)

        images.append(x)
        keypoints_1d.append(img_keypoints)
        keypoints_2d.append(x_condition)

    return images, keypoints_1d, keypoints_2d


def generate_keypoint_condition(kps, opt):
    a_path_rgb = np.zeros((opt.image_size_y, opt.image_size_x, 3))
    colors = keypoint_functions.get_keypoint_colors()
    keypoint_layers = keypoint_functions.load_layer_information(opt)

    kps_2d = keypoint_functions.create_keypoint_condition(a_path_rgb, kps, opt, num_keypoints=opt.num_keypoints)
    kps_2d = torch.from_numpy(kps_2d)
    kps_2d = (kps_2d + 1) / 2.0

    # each keypoint condition for an image is now a list where each list contains the information
    # about the keypoints in the given layer for the given image
    layered_keypoints_2d = []
    for layer in keypoint_layers:
        layered_keypoints_2d.append(kps_2d[[layer], :, :].squeeze())
    kps_2d = layered_keypoints_2d

    layered_keypoints_1d = []
    for layer in keypoint_layers:
        current_keypoint_1d = {x: kps[x] for x in layer}
        layered_keypoints_1d.append(current_keypoint_1d)
    kps = layered_keypoints_1d

    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    img_label = []

    for layer_idx in range(len(keypoint_layers)):
        a_path_rgb = np.zeros((3, opt.image_size_y, opt.image_size_x))
        for idx in range(kps_2d[layer_idx].shape[0]):
            current_kp = np.expand_dims(kps_2d[layer_idx][idx], 0)
            current_color = np.zeros_like(a_path_rgb)
            current_color[0] = colors[idx][0]
            current_color[1] = colors[idx][1]
            current_color[2] = colors[idx][2]
            a_path_rgb = a_path_rgb + (np.repeat(current_kp, repeats=3, axis=0) * current_color)

            if opt.skeleton:
                skeleton = keypoint_functions.load_skeleton_info(opt)
                a_path_rgb = keypoint_functions.add_skeleton(a_path_rgb, kps[layer_idx], skeleton, opt)

        img = a_path_rgb * 255.
        img = img.astype(np.uint8)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # img.save("kp_{}.jpg".format(layer_idx))
        img_label.append(transform(img).unsqueeze(0).to(opt.device))
    # exit()

    return img_label


def load_model(opt):
    from models import networks
    def load_network(network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('{} does not exist!'.format(save_path))
            exit()
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)
        return network

    netG = networks.define_G(opt.input_nc, opt.output_nc, opt, opt.ngf, norm=opt.norm, gpu_ids=opt.gpu_ids)
    netG = load_network(netG, 'G', opt.which_epoch, opt.name)
    return netG


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
