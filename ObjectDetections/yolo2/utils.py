"""
Help functions for YOLOv2
"""
import random
import colorsys

import cv2
import numpy as np



############## preprocess image ##################


def preprocess_image(image, image_size=(416, 416)):
    """Preprocess a image to inference"""
    image_cp = np.copy(image).astype(np.float32)
    # resize the image
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    # normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    # expand the batch_size dim
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

def postprocess(bboxes, obj_probs, class_probs, image_shape=(416, 416),
                threshold=0.5):
    """post process the detection results"""
    bboxes = np.reshape(bboxes, [-1, 4])
    bboxes[:, 0::2] *= float(image_shape[1])
    bboxes[:, 1::2] *= float(image_shape[0])
    bboxes = bboxes.astype(np.int32)

    # clip the bboxs
    bbox_ref = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    bboxes = bboxes_clip(bbox_ref, bboxes)

    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_inds = np.argmax(class_probs, axis=1)
    class_probs = class_probs[np.arange(len(obj_probs)), class_inds]
    scores = obj_probs * class_probs

    # filter bboxes with scores > threshold
    keep_inds = scores > threshold
    bboxes = bboxes[keep_inds]
    scores = scores[keep_inds]
    class_inds = class_inds[keep_inds]

    # sort top K
    class_inds, scores, bboxes = bboxes_sort(class_inds, scores, bboxes)
    # nms
    class_inds, scores, bboxes = bboxes_nms(class_inds, scores, bboxes)

    return bboxes, scores, class_inds

def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # for display
    ############################
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.)
                  for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(imgcv, mess, text_loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv


############## process bboxes ##################
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]






