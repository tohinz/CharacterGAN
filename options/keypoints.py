# Watercolor-Man and Watercolor-Lady
__keypoint_labels_watercolor = {
        "Head": 0,
        "Left-Hand": 1,
        "Right-Hand": 2,
        "Left-Elbow": 3,
        "Right-Elbow": 4,
        "Torso": 5,
        "Left-Knee": 6,
        "Right-Knee": 7,
        "Left-Foot": 8,
        "Right-Foot": 9,
        "Left-Shoulder": 10,
        "Right-Shoulder": 11,
        "Left-Hip": 12,
        "Right-Hip": 13,
        "Tail-1": 14,
        "Tail-2": 15,
    }

# Fox
__keypoint_labels_fox = {
        "Head": 0,
        "Left-Hand": 1,
        "Right-Hand": 2,
        "Left-Elbow": 3,
        "Right-Elbow": 4,
        "Torso": 5,
        "Left-Knee": 6,
        "Right-Knee": 7,
        "Left-Foot-1": 8,
        "Right-Foot-1": 9,
        "Left-Shoulder": 10,
        "Right-Shoulder": 11,
        "Left-Hip": 12,
        "Right-Hip": 13,
        "Tail-1": 14,
        "Left-Foot-2": 15,
        "Right-Foot-2": 16,
    }

# Dino
__keypoint_labels_dino = {
        "Head": 0,
        "Left-Hand": 1,
        "Right-Hand": 2,
        "Right-Elbow": 3,
        "Torso": 4,
        "Left-Knee": 5,
        "Right-Knee": 6,
        "Left-Foot": 7,
        "Right-Foot": 8,
        "Right-Shoulder": 9,
        "Left-Hip": 10,
        "Right-Hip": 11,
        "Tail-1": 12,
        "Tail-2": 13,
    }


# randomly chosen RGB colors for keypoints
# yes, hardcoded!
# add more if you have more than 16 keypoints
__keypoint_colors = {
            0: [.3, 0, 0],
            1: [0, .3, 0],
            2: [0, 0, .3],
            3: [.3, .3, 0],
            4: [0, .3, .3],
            5: [.3, 0, .3],
            6: [.3, .3, .3],
            7: [.1, .3, .4],
            8: [.4, .1, .3],
            9: [.3, .4, .1],
            10: [.1, .3, .1],
            11: [.1, .1, .3],
            12: [.3, .1, .1],
            13: [.1, .1, 0],
            14: [.1, 0, .1],
            15: [.1, .1, .1]
        }


__keypoint_labels_inverse_watercolor = {v: k for k, v in __keypoint_labels_watercolor.items()}
__keypoint_labels_inverse_dino = {v: k for k, v in __keypoint_labels_dino.items()}
__keypoint_labels_inverse_fox = {v: k for k, v in __keypoint_labels_fox.items()}


def get_keypoints(opt):
    if "watercolor" in opt.dataroot.lower():
        return __keypoint_labels_watercolor
    elif "dino" in opt.dataroot.lower():
        return __keypoint_labels_dino
    elif "fox" in opt.dataroot.lower():
        return __keypoint_labels_fox
    else:
        print("Keypoints for dataset {} not found.")
        print("Please update file \"options/keypoints.py\"")
        exit()


def get_keypoints_label(opt):
    if "watercolor" in opt.dataroot.lower():
        return __keypoint_labels_inverse_watercolor
    elif "dino" in opt.dataroot.lower():
        return __keypoint_labels_inverse_dino
    elif "fox" in opt.dataroot.lower():
        return __keypoint_labels_inverse_fox
    else:
        print("Keypoints for dataset {} not found.")
        print("Please update file \"options/keypoints.py\"")
        exit()


def get_keypoint_colors():
    return __keypoint_colors