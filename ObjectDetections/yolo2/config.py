"""
Yolov2 anchors and coco classes
"""

"""
anchors = [[0.738768, 0.874946],
           [2.42204, 2.65704],
           [4.30971, 7.04493],
           [10.246, 4.59428],
           [12.6868, 11.8741]]
"""
anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

def read_coco_labels():
    f = open("./data/coco_classes.txt")
    class_names = []
    for l in f.readlines():
        class_names.append(l[:-1])
    return class_names

class_names = read_coco_labels()