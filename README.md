# CharacterGAN

Implementation of the paper [*"CharacterGAN: Few-Shot Keypoint Character Animation and Reposing"* by Tobias Hinz, Matthew Fisher, Oliver Wang, Eli Shechtman, and Stefan Wermter](google.com) (open with Adobe Acrobat or similar to see visualizations). Supplementary material can be found [supplementary/
CharacterGAN_Supplementary.pdf](google.com).

Our model can be trained on only a *few images (e.g. 10) of a given character* labeled with user-chosen *keypoints*.
The resulting model can be used to *animate* the character on which it was trained by interpolating between its poses specified by their keypoints.
We can also *repose* characters by simply moving the keypoints into the desired positions.
To train the model all we need are few images depicting the character in diverse poses from the same viewpoint, keypoints, a file that describes how the keypoints are connected (the characters *skeleton*) and which keypoints lie in the same layer.

### Examples

**Animation**: For all examples the model was trained on 8-15 images (see first row) of the given character.

Training Images |  12         |  15          |  9         |  12          |  15         |  15          |  8
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Animation |  ![dog_animation](gifs/dog.gif) |  ![maddy_animation](gifs/maddy.gif) |  ![ostrich_animation](gifs/ostrich.gif) |  ![man_animation](gifs/stock_man.gif) |  ![robot_animation](gifs/evans.gif) |  ![man_animation](gifs/watercolor_man.gif) |  ![cow_animation](gifs/cow.gif)

<br/><br/>

**Frame interpolation**: Example of interpolations between two poses with the start and end keypoints highlighted.<br />

![man](interpolations/Man/kp_pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0001.jpg) |   ![man](interpolations/Man/pm_gen_img_0002.jpg) |   ![man](interpolations/Man/pm_gen_img_0003.jpg) |   ![man](interpolations/Man/pm_gen_img_0004.jpg) |   ![man](interpolations/Man/pm_gen_img_0005.jpg) |   ![man](interpolations/Man/pm_gen_img_0006.jpg) |   ![man](interpolations/Man/pm_gen_img_0007.jpg) |   ![man](interpolations/Man/pm_gen_img_0008.jpg) |   ![man](interpolations/Man/pm_gen_img_0009.jpg) |  ![man](interpolations/Man/pm_gen_img_0010.jpg)  |   ![man](interpolations/Man/kp_pm_gen_img_0010.jpg)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![dog](interpolations/Dog/kp_pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0001.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0002.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0003.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0004.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0005.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0006.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0007.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0008.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0009.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0010.jpg)  |   ![dog](interpolations/Dog/kp_pm_gen_img_0010.jpg)

<br/><br/>

**Reposing**: You can use our interactive GUI to easily repose a given character based on keypoints.

Interactive |![dog_gui](gifs/gui_dog.gif) | ![man_gui](gifs/gui_stock_man.gif)
:-------------------------:|:-------------------------:|:-------------------------:
**Gui** | ![cow_gui](gifs/gui_cow.gif) | ![man_gui](gifs/gui_watercolor_man.gif)

<br/>

## Installation

- python 3.8
- pytorch 1.7.1

```
pip install -r requirements.txt
```

## Training

### Training Data
All training data for a given character should be in a single folder.
We used [this website](https://www.makesense.ai/) to label our images but there are of course other possibilities.

The folder should contain:
* all training images (all in the same resolution),
* a file called `keypoints.csv` (containing the keypoints for each image),
* a file called `keypoints_skeleton.csv` (containing skeleton information, i.e. how keypoints are connected with each other), and
* a file called `keypoints_layers.csv` (containing the information about which layer each keypoint resides in).

The structure of the `keypoints.csv` file is (no header): `keypoint_label,x_coord,y_coord,file_name`.
The first column describes the keypoint label (e.g. *head*), the next two columns give the location of the keypoint, and the final column states which training image this keypoint belongs to.

The structure of the `keypoints_skeleton.csv` file is (no header): `keypoint,connected_keypoint,connected_keypoint,...`.
The first column describes which keypoint we are describing in this line, the following columns describe which keypoints are connected to that keypoint (e.g. *elbow, shoulder, hand* would state that the *elbow* keypoint should be connected to the *shoulder* keypoint and the *hand* keypoint).

The structure of the `keypoints_layers.csv` file is (no header): `keypoint,layer`.
"Keypoint" is the keypoint label (same as used in the previous two files) and "layer" is an integer value desribing which layer the keypoint resides in.

See our example training data in `datasets` for examples of both files.

We provide two examples (produced by [Zuzana Studená](https://artemisfilm.webnode.com/)) for training, located in `datasets`.
Our other examples were trained on data from [Adobe Stock](https://stock.adobe.com/) or from [Character Animator](https://www.adobe.com/products/character-animator.html) and I currently have no license to distribute them.
You can purchase the Stock data here:
* [Man](https://stock.adobe.com/images/running-man-cycle-of-animation-for-men-s-running-2d-cartoon-character/325376414): we used all images
* [Dog](https://stock.adobe.com/images/great-dane-dog-run-cycle-animation-sequence/292133555): we used all images
* [Ostrich](https://stock.adobe.com/images/ostrich-run-cycle-animation-sequence-loop-animation-sprite-sheet/326702824): we used the first nine images
* [Cow](https://stock.adobe.com/images/cow-walk-cycle-animation-frames-loop-animation-sequence-sprite-sheet/326705734): we used the first eight images

There are also several websites where you can download Sprite sheets for free.

### Train a Model
To train a model with the default parameters from our paper run:

```
python train.py --gpu_ids 0 --num_keypoints 14 --dataroot datasets/Watercolor-Man --fp16 --name Watercolor-Man
```

Training one model should take about 60 (FP16) to 90 (FP32) minutes on an NVIDIA GeForce GTX 2080Ti.
You can usually use fewer iterations for training and still achieve good results (see next section).

### Training Parameters
You can adjust several parameters at train time to possibly improve your results.

* `--name` to change the name of the folder in which the results are stored (default is `CharacterGAN-Timestamp`)
* `--niter 4000` and `--niter_decay 4000` to adjust the number of training steps (`niter_decay`is the number of training steps during which we reduce the learning rate linearly; default is 8000 for both, but you can get good results with fewer iterations)
* `--mask True --output_nc 4` to train with a mask
* `--skeleton False` to train without skeleton information
* `--bkg_color 0` to set the background color of the training images to *black* (default is white, only important if you train with a mask)
* `--batch_size 10` to train with a different batch size (default is 5)

The file `options/keypoints.py` lets you modify/add/remove keypoints for your characters.

### Results
The output is saved to `checkpoints/` and we log the training process with Tensorboard.
To monitor the progress go to the respective folder and run

```
 tensorboard --logdir .
```

# Testing
At test time you can either use the model to animate the character or use our interactive GUI to change the position of individual keypoints.

### Animate Character
To animate a character (or create interpolations between two images):

```
python animate_example.py --gpu_ids 0 --model_path checkpoints/Watercolor-Man-.../ --img_animation_list datasets/Watercolor-Man/animation_list.txt --dataroot datasets/Watercolor-Man
```

`--img_animation_list` points to a file that lists the images that should be used for animation. The file should contain one file name per line pointing to an image in `dataroot`. The model then generates an animation by interpolating between the images in the given order. See `datasets/Watercolor-Man/animation_list.txt` for an example.

You can add `--draw_kps` to visualize the keypoints in the animation.
You can specifiy the gif parameters by setting `--num_interpolations 10` and `--fps 5`.
`num_interpolations` specifies how many images are generated between two real images (from `img_animation_list`), `fps` determines the frames per second of the generated gif.

### Modify Individual Keypoints
To run the interactive GUI:

```
python visualizer.py --gpu_ids 0 --model_path checkpoints/Watercolor-Man-.../
```

Set `--gpu_ids -1` to run the model on a CPU.
You can also scale the images during visualization, e.g. use `--scale 2`.

## Patch-based Refinement
We use [this implementation](https://github.com/jamriska/ebsynth) to run the patch-based refinement step on our generated images.
The easiest way to do this is to merge all your training images into a single large image file and use this image file as the style and source image.

## Acknowledgements
Our implementation uses code from [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD), the TPS augmentation from [DeepSIM](https://github.com/eliahuhorwitz/DeepSIM), and the patch-based refinement code from [https://ebsynth.com/](https://ebsynth.com/) ([GitHub](https://github.com/jamriska/ebsynth)).

We would also like to thank [Zuzana Studená](https://artemisfilm.webnode.com/) who produced some of the artwork used in this work.

## Citation
If you found this code useful please consider citing:

```
@article{hinz2021improved,
    author    = {Hinz, Tobias and Fisher, Matthew and Wang, Oliver and Shechtman, Eli and Wermter, Stefan},
    title     = {CharacterGAN: Few-Shot Keypoint Character Animation and Reposing},
    journal = {ArXiV},
    year      = {2021}
}
```
