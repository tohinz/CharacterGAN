# CharacterGAN

Implementation of the paper [*"CharacterGAN: Few-Shot Keypoint Character Animation and Reposing"* by Tobias Hinz, Matthew Fisher, Oliver Wang, Eli Shechtman, and Stefan Wermter](google.com) (open with Adobe Acrobat or similar to see visualizations). Supplementary material can be found [here](google.com).

Our model can be trained on only a *few images (e.g. 10) of a given character* labeled with user-chosen *keypoints*.
The resulting model can be used to *animate* the character on which it was trained by interpolating between its poses specified by their keypoints.
We can also *repose* characters by simply moving the keypoints into the desired positions.
To train the model all we need are few images depicting the character in diverse poses from the same viewpoint, keypoints, and a file that describes how the keypoints are connected (the characters *skeleton*).

### Examples

**Animation**: For all examples the model was trained on 8-15 images (see first row) of the given character.

Training Images |  12         |  15          |  9         |  12          |  15         |  15          |  8
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Animation |  ![dog_animation](gifs/dog.gif) |  ![maddy_animation](gifs/maddy.gif) |  ![ostrich_animation](gifs/ostrich.gif) |  ![man_animation](gifs/stock_man.gif) |  ![robot_animation](gifs/evans.gif) |  ![man_animation](gifs/watercolor_man.gif) |  ![cow_animation](gifs/cow.gif)

**Frame interpolation**: Example of interpolations between two poses with the start and end keypoints highlighted:

![man](interpolations/Man/kp_pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0001.jpg) |   ![man](interpolations/Man/pm_gen_img_0002.jpg) |   ![man](interpolations/Man/pm_gen_img_0003.jpg) |   ![man](interpolations/Man/pm_gen_img_0004.jpg) |   ![man](interpolations/Man/pm_gen_img_0005.jpg) |   ![man](interpolations/Man/pm_gen_img_0006.jpg) |   ![man](interpolations/Man/pm_gen_img_0007.jpg) |   ![man](interpolations/Man/pm_gen_img_0008.jpg) |   ![man](interpolations/Man/pm_gen_img_0009.jpg) |  ![man](interpolations/Man/pm_gen_img_0010.jpg)  |   ![man](interpolations/Man/kp_pm_gen_img_0010.jpg)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![dog](interpolations/Dog/kp_pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0001.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0002.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0003.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0004.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0005.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0006.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0007.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0008.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0009.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0010.jpg)  |   ![dog](interpolations/Dog/kp_pm_gen_img_0010.jpg)

**Reposing**: You can use our interactive GUI to easily repose a given character based on keypoints.
TODO

## Installation

- python 3.8
- pytorch 1.7.1

```
pip install -r requirements.txt
```

## Requirements
Images (we do 250px, larger possible I guess)
Keypoints
Skeleton

## Training

### Training Data
All training data for a given character should be in a single folder.
Concretely: the folder should contain all training images (all in the same resolution), a file called `keypoints.csv`, and (potentially) a file called `keypoints_skeleton.csv`. We used [this website](https://www.makesense.ai/) to label our images but there are of course other possibilities.

The structure of the `keypoints.csv` file is (no header): `keypoint_label,x_coord,y_coord,file_name`.
The first column describes the keypoint label (e.g. *head*), the next two columns give the location of the keypoint, and the final column states which training image this keypoint belongs to.

The structure of the `keypoints_skeleton.csv` file is (no header): `keypoint,connected_keypoint,connected_keypoint,...`.
The first column describes which keypoint we are describing in this line, the following columns describe which keypoints are connected to that keypoint (e.g. *elbow, shoulder, hand* would state that the *elbow* keypoint should be connected to the *shoulder* keypoint and the *hand* keypoint).

See our example training data in `datasets` for examples of both files.


To train a model with the default parameters from our paper run:

```
python train.py --gpu_ids 0 
```

Training one model should take about 60 (FP16) to 80 (FP32) minutes on an NVIDIA GeForce GTX 2080Ti.

### Parameters to Modify
image size
mask




```
python train.py --gpu_ids 0
```


### Results
The output is saved to `checkpoints/` and we log the training process with Tensorboard.
To monitor the progress go to the respective folder and run

```
 tensorboard --logdir .
```

# Testing
At test time you can either use the model to animate the character or use our interactive GUI to change the position of individual keypoints.

### Animate Character
animate

```
python 
```


### Modify Individual Keypoints
GUI

```
python 
```


## Patch-based refinement
We use [this implementation](https://github.com/jamriska/ebsynth) to run the patch-based refinement step on our generated images.
The easiest way to do this is to merge all your training images into a single large image file and use this image file as the style and source image.

## Acknowledgements
Our implementation uses code from [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD), the TPS augmentation from [DeepSIM](https://github.com/eliahuhorwitz/DeepSIM), and the patch-based refinement code from [https://ebsynth.com/](https://ebsynth.com/) ([GitHub](https://github.com/jamriska/ebsynth)).

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
