import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import datetime
from collections import OrderedDict
import dateutil.tz
from tqdm import tqdm
from skimage import io as img

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from options.train_options import TrainOptions
from data.dataset import KPDataset
from models.pix2pixHD_model import Pix2PixHDModel


# parse options
opt = TrainOptions().parse()
opt.name = opt.name + "-" + datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
opt.dir2save = os.path.join(opt.checkpoints_dir, opt.name)
writer = SummaryWriter(log_dir=opt.dir2save)

# create output folder
try:
    os.makedirs(opt.dir2save)
except OSError:
    pass

# get resolution of training images
_training_imgs = glob.glob(os.path.join(opt.dataroot, "*.jpg")) + glob.glob(os.path.join(opt.dataroot, "*.png")) + glob.glob(os.path.join(opt.dataroot, "*.jpeg"))
_img = img.imread(_training_imgs[0])
opt.image_size_x = _img.shape[0]
opt.image_size_y = _img.shape[1]

# save training parameters and files
with open(osp.join(opt.dir2save, 'parameters.txt'), 'w') as f:
    for o in opt.__dict__:
        f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
current_path = os.path.dirname(os.path.abspath(__file__))
for py_file in glob.glob(osp.join(current_path, "*.py")):
    copyfile(py_file, osp.join(opt.dir2save, py_file.split("/")[-1]))
copytree(osp.join(current_path, "data"), osp.join(opt.dir2save, "data"))
copytree(osp.join(current_path, "models"), osp.join(opt.dir2save, "models"))
copytree(osp.join(current_path, "options"), osp.join(opt.dir2save, "options"))
copytree(osp.join(current_path, "util"), osp.join(opt.dir2save, "util"))

torch.backends.cudnn.benchmark = len(opt.gpu_ids) > 0  # might lead to OOM during training

# initialize models
model = Pix2PixHDModel()
model.initialize(opt)
if len(opt.gpu_ids):
    model.cuda()
if opt.fp16:
    import torch.cuda.amp as amp
    scaler = amp.GradScaler()
optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D
print("Initialize models...done")

# initialize data set
dataset = KPDataset()
dataset.initialize(opt)
dataloader = iter(dataset.createInfiniteLoader())
print("Initialize data set...done ({} training images)".format(len(dataset)))
print("Start training. Saving to: {}".format(opt.dir2save))

# start training
for iter in tqdm(range(0, opt.niter + opt.niter_decay)):
    data = next(dataloader)

    ############## Forward Pass ######################
    if opt.fp16:
        with amp.autocast():
            losses, generated = model(data['keypoint'], data['image'])
    else:
        losses, generated = model(data['keypoint'], data['image'])

    loss_D = (losses['D_fake'] + losses['D_real']) * 0.5
    loss_G = losses['G_GAN'] + losses.get('G_GAN_Feat', 0) + losses.get('G_VGG', 0)


    ############### Backward Pass ####################
    # update generator weights
    optimizer_G.zero_grad()
    if opt.fp16:
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
    else:
        loss_G.backward()
        optimizer_G.step()

    # update discriminator weights
    optimizer_D.zero_grad()
    if opt.fp16:
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()
    else:
        loss_D.backward()
        optimizer_D.step()

    ############## Log progress ##########
    if iter % opt.logging_freq == 0 or iter + 1 == opt.niter + opt.niter_decay:

        # log generated images and corresponding keypoints
        real_images = data['image']
        grid = make_grid(real_images, nrow=min(5, opt.batch_size), normalize=True, range=(-1, 1))
        writer.add_image('real_images', grid, iter)

        generated_images = generated.data
        grid = make_grid(generated_images, nrow=min(5, opt.batch_size), normalize=True, range=(-1, 1))
        writer.add_image('generated_images', grid, iter)

        keypoints = data['keypoint']
        for layer in range(opt.num_kp_layers):
            grid = make_grid(keypoints[layer], nrow=min(5, opt.batch_size), normalize=True, range=(-1, 1))
            writer.add_image('keypoints_{}'.format(layer), grid, iter)

        # log loss values
        writer.add_scalar('Loss/D/real', losses['D_real'].item(), iter)
        writer.add_scalar('Loss/D/fake', losses['D_fake'].item(), iter)
        writer.add_scalar('Loss/G/adv_loss', losses['G_GAN'].item(), iter)
        writer.add_scalar('Loss/G/vgg_loss', losses['G_VGG'].item(), iter)
        writer.add_scalar('Loss/G/fmm_loss', losses['G_GAN_Feat'].item(), iter)

        # save model
        model.save('latest')

    ############## Reduce learning rate ##########
    # linearly decay learning rate after certain iterations
    if iter > opt.niter:
        model.update_learning_rate()

writer.close()
