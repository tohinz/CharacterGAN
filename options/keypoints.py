__keypoint_labels = {
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


__keypoint_labels_inverse = {v: k for k, v in __keypoint_labels.items()}

def get_keypoints():
    return __keypoint_labels


def get_keypoints_label():
    return __keypoint_labels_inverse


def get_keypoint_colors():
    return __keypoint_colors