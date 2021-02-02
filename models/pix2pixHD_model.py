import os
import torch

from . import networks

class Pix2PixHDModel(torch.nn.Module):
    def name(self):
        return 'Pix2PixHDModel'
    
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.isTrain = opt.isTrain
        input_nc = opt.input_nc
        self.upsample = torch.nn.Upsample(size=[opt.image_size_y, opt.image_size_x], mode='nearest')

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.use_cuda = len(self.gpu_ids) > 0

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt, opt.ngf, norm=opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            netD_input_nc = opt.num_kp_layers * input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                          num_D=opt.num_D, gpu_ids=self.gpu_ids)

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss()

            # initialize optimizers
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, label, image):
        # Prepare inputs
        input_label = [l.detach() for l in label]
        real_image = image.detach()
        if self.use_cuda:
            input_label = [l.cuda() for l in input_label]
            real_image = real_image.cuda()

        # Fake Generation
        fake_image = self.netG.forward(input_label)
        if fake_image.shape[2] != real_image.shape[2]:
            fake_image = self.upsample(fake_image)

        # Fake Detection and Loss
        input_label = torch.cat(input_label, 1)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1).detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real Detection and Loss        
        pred_real = self.netD.forward(torch.cat((input_label, real_image), dim=1).detach())
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(fake_image[:, :3, :, :], real_image[:, :3, :, :]) * self.opt.lambda_feat

        return {"G_GAN": loss_G_GAN, "G_GAN_Feat": loss_G_GAN_Feat, "G_VGG": loss_G_VGG,
                "D_real": loss_D_real, "D_fake": loss_D_fake}, fake_image

    def inference(self, label):
        input_label = [l.detach() for l in label]
        if self.use_cuda:
            input_label = [l.cuda() for l in input_label]

        with torch.no_grad():
            fake_image = self.netG.forward(input_label)

        return fake_image

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr
