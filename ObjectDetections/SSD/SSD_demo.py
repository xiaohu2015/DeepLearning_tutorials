"""
SSD demo
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from ssd_300_vgg import SSD
from utils import preprocess_image, process_bboxes
from visualization import plt_bboxes



ssd_net = SSD()
classes, scores, bboxes = ssd_net.detections()
images = ssd_net.images()

sess = tf.Session()
# Restore SSD model.
ckpt_filename = './ssd_checkpoints/ssd_vgg_300_weights.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

img = cv2.imread('./demo/dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_prepocessed = preprocess_image(img)

rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes],
                                      feed_dict={images: img_prepocessed})


rclasses, rscores, rbboxes = process_bboxes(rclasses, rscores, rbboxes)

plt_bboxes(img, rclasses, rscores, rbboxes)



