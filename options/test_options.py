from .base_options import BaseOptions
from .base_options import str2bool

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for evaluation in general
        self.parser.add_argument('--model_path', help='path_to_model', required=True)

        # for animation
        self.parser.add_argument('--img_animation_list', type=str, help='list of image order for animation', default="")
        self.parser.add_argument('--draw_kps', type=str2bool, help='draw keypoints on animation', default=False)
        self.parser.add_argument('--num_interpolations', type=int, default=10, help='interpolation frames between keypoints')
        self.parser.add_argument('--fps', type=int, default=5, help='fps of generated gif')
        self.parser.add_argument('--batch_size', type=int, default=50, help='batch size during generation')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to use')

        # for interactive GUI
        self.parser.add_argument('--scale', type=int, default=1, help='scale image')


        self.isTrain = False
