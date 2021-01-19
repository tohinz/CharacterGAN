# CharacterGAN

Official implementation of the paper [*"CharacterGAN: Few-Shot Keypoint Character Animation and Reposing"* by Tobias Hinz, Matthew Fisher, Oliver Wang, Eli Shechtman, and Stefan Wermter](google.com) (open with Adobe Acrobat or similar to see visualizations).

[Supplementary Material](google.com).

Our model can be trained on only a few images (e.g. 10) of a given character labeled with user-chosen keypoints.
The resulting model can be used to animate or repose the character.


Examples of animation.
For all examples the model was trained on 8-15 images (see first row) of the given character.

12         |  15          |  9         |  12          |  15         |  15          |  8
:-------------------------:|:-------------------------:|:-------------------------:
![dog_animation](gifs/dog.gif) |  ![maddy_animation](gifs/maddy.gif) |  ![ostrich_animation](gifs/ostrich.gif) |  ![man_animation](gifs/man.gif) |  ![robot_animation](gifs/evans.gif) |  ![man_animation](gifs/watercolor_man.gif) |  ![cow_animation](gifs/cow.gif)

Example of interpolations between two poses with the start and end keypoints highlighted:

![man](interpolations/Man/kp_pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0000.jpg) |  ![man](interpolations/Man/pm_gen_img_0001.jpg) |   ![man](interpolations/Man/pm_gen_img_0002.jpg) |   ![man](interpolations/Man/pm_gen_img_0003.jpg) |   ![man](interpolations/Man/pm_gen_img_0004.jpg) |   ![man](interpolations/Man/pm_gen_img_0005.jpg) |   ![man](interpolations/Man/pm_gen_img_0006.jpg) |   ![man](interpolations/Man/pm_gen_img_0007.jpg) |   ![man](interpolations/Man/pm_gen_img_0008.jpg) |   ![man](interpolations/Man/pm_gen_img_0009.jpg) |  ![man](interpolations/Man/pm_gen_img_0010.jpg)  |   ![man](interpolations/Man/kp_pm_gen_img_0010.jpg)
:-------------------------:|:-------------------------:|:-------------------------:
![dog](interpolations/Dog/kp_pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0000.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0001.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0002.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0003.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0004.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0005.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0006.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0007.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0008.jpg) |   ![dog](interpolations/Dog/pm_gen_img_0009.jpg) |  ![dog](interpolations/Dog/pm_gen_img_0010.jpg)  |   ![dog](interpolations/Dog/kp_pm_gen_img_0010.jpg)

# Installation

- python 3.8
- pytorch 1.7.1

```
pip install -r requirements.txt
```

# Requirements
Images (we do 250px, larger possible I guess)
Keypoints
Skeleton

# Training
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



# Acknowledgements
Our implementation uses code from [Pix2PixHD](google.com) and the TPS augmentation from [DeepSIM](google.com).

# Citation
If you found this code useful please consider citing:

```
@article{hinz2021improved,
    author    = {Hinz, Tobias and Fisher, Matthew and Wang, Oliver and Shechtman, Eli and Wermter, Stefan},
    title     = {CharacterGAN: Few-Shot Keypoint Character Animation and Reposing},
    journal = {ArXiV},
    year      = {2021}
}
```
