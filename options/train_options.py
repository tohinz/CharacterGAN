from .base_options import BaseOptions
from .base_options import str2bool

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for logging
        self.parser.add_argument('--logging_freq', type=int, default=500, help='frequency of saving results')

        # for training
        self.parser.add_argument('--niter', type=int, default=8000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=8000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--batch_size', type=int, default=5, help='batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='workers for dataloader')
        self.parser.add_argument('--tps_aug', type=str2bool, default=True, help='apply tps augmentations during training')
        self.parser.add_argument('--tps_points_per_dim', type=int, default=3)
        self.parser.add_argument('--tps_scale', type=float, default=0.1, help='stretch scale for TPS')
        self.parser.add_argument('--bkg_color', type=int, default=255, help='background color (default is white)')

        self.isTrain = True
