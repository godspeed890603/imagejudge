import numpy as np
from matplotlib import pyplot as plt

from bounding_box_utils.bounding_box_utils import iou

from myobject_detection_2d_data_generator import DataGenerator
from transformations.object_detection_2d_patch_sampling_ops import *
from transformations.object_detection_2d_geometric_ops import *
from transformations.object_detection_2d_photometric_ops import *
from transformations.object_detection_2d_image_boxes_validation_utils import *
from data_augmentation_chains.data_augmentation_chain_original_ssd import *
import cv2


# dataset = DataGenerator(labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'))

dataset = DataGenerator()
images_dir         = 'tutorial_dataset/JPEGImages/'
# annotations_dir    = 'tutorial_dataset/Annotations/'
image_set_filename = 'tutorial_dataset/ImageSets/imageset.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor']

classes = []

# dataset.parse_xml(images_dirs=[images_dir],
#                   image_set_filenames=[image_set_filename],
#                   annotations_dirs=[annotations_dir],
#                   classes=classes,
#                   include_classes='all',
#                   exclude_truncated=False,
#                   exclude_difficult=False,
#                   ret=False)
print("generate1-2")
dataset.parse_xml(images_dirs=[images_dir],
                  image_set_filenames=[image_set_filename],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)
# # image_set_filename="D:/temp/imagejudge/data_generator_object_detection_2d-master/tutorial_dataset/JPEGImages/000030.JPG"
# image_set_filename="D:/8443202315719.jpg"
# dataset.set_image(image_set_filename)


print("generate1-1")
# translate = Translate(dy=-0.2,
#                       dx=0.3,
#                       clip_boxes=False,
#                       box_filter=None,
#                       background=(0,0,0))

translate = Translate(dy=0,
                      dx=0,
                      clip_boxes=False,
                      box_filter=None,
                      background=(0,0,0))

batch_size = 2
print("generate1")
data_generator = dataset.generate(batch_size=batch_size,
                                  shuffle=False,
                                  transformations=[translate],
                                  label_encoder=None,
                                  returns={'processed_images'},
                                  keep_images_without_gt=False)

print("generate2")
# data_generator = dataset.generate(batch_size=batch_size,
#                                   shuffle=False,
#                                   transformations=[translate],
#                                   label_encoder=None,
#                                   returns={'processed_images',
#                                            'processed_labels',
#                                            'filenames',
#                                            'original_images',
#                                            'original_labels'},
#                                   keep_images_without_gt=False)


# 
# 
# # processed_images, processed_annotations, filenames, original_images, original_annotations = next(data_generator)
processed_images= next(data_generator)
# processed_images= data_generator.next()

print(len(processed_images))
print(type(processed_images))

print(len(processed_images))
print(type(processed_images[0]))

# print(processed_images)
# 
# i = 0 # Which batch item to look at
# 
# print("Image:", filenames[i])
# print()
# print("Original ground truth boxes:\n")
# print(np.array(processed_images[i]))
# 
# 
#
print(type(processed_images))
# print(processed_images) 
cv2.imshow("outsidetest", processed_images[0])
  
cv2.waitKey(0)
cv2.destroyAllWindows()

