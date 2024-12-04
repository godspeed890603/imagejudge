from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from ssd_utils import BBoxUtility
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
import cv2
import os
from os.path import basename

# ~~~~~~~~~~~load image~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~
img_height = 375 # Height of the input images
img_width = 500 # Width of the input images


K.clear_session() # Clear previous models from memory.




# ~~~~~~~~~read Image~~~~~~~~

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


# image_set_filename="D:/temp/imagejudge/data_generator_object_detection_2d-master/tutorial_dataset/JPEGImages/000030.JPG"
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
defectImage=processed_images[0]
defectImage1=defectImage.reshape(1,defectImage.shape[0],defectImage.shape[1],3)
# 
# # channels_first: ��m�q�D(R/G/B)���(�`��)��b��2���סA��3�B4���ש�m�e�P��
# if K.image_data_format() == 'channels_first':
#     defectImage = defectImage.reshape( 1, img_width, img_height,3)
# #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
# #     input_shape = (1, img_rows, img_cols)
#     print("channels_first")
# else: # channels_last: ��m�q�D(R/G/B)���(�`��)��b��4���סA��2�B3���ש�m�e�P��
#     defectImage = defectImage.reshape( 1,img_width, img_height, 3)
# #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# #     input_shape = (img_rows, img_cols, 1)
#     print("channels_last")




# 
# i = 0 # Which batch item to look at
 
# print("Image:", filenames[i])
# print()
# print("Original ground truth boxes:\n")
# print(np.array(processed_images[i]))
#  
#  
#
print(processed_images) 
cv2.moveWindow("pre",1000,100)
cv2.imshow("pre", defectImage)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

# ~~~~~~~~~~read image end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




































##img_height = 300 # Height of the input images
##img_width = 480 # Width of the input images
# img_height = 375 # Height of the input images
# img_width = 500 # Width of the input images

img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size




# 1: Build the Keras model

# K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Optional: Load some weights

model.load_weights('./ssd7_model1.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)







# 3: Make a prediction

##image = cv2.imread("D:/temp/cv/ssd_keras-master/datasets/udacity_driving_datasets/UUUUUU.jpg",1)
# image = cv2.imread("D:/temp/cv/ssd_keras-master/datasets/udacity_driving_datasets/U3-1.JPG",1)
# image = cv2.imread("D:/temp/imagejudge/data_generator_object_detection_2d-master/tutorial_dataset/JPEGImages/000030.JPG",1)
# print(image)
# imgPaath="D:/temp/cv/ssd_keras-master/datasets/udacity_driving_datasets/1.JPG"
# image = cv2.imread(imgPaath,1)


# test111= DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None,filenames=imgPaath,
#                  filenames_type='text',)



# la=list()
# la.append(image)
# 
# batch_images1=np.array(la)
# 
# # ~~kermit opencv show Start~~~
# cv2.imshow("win", batch_images1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ~~kermit opencv show Start~~~
 # 頛詨鞈���蔣�����
print("影像預先處理.......")
# inputs = preprocess_input(np.array(batch_images1))

# batch_images=np.array(batch_images1)#batch_images1


# cv2.imshow("win", batch_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



##y_pred = model.predict(image)
# y_pred = model.predict(batch_images)
y_pred = model.predict(defectImage1)

print(y_pred)


#  # create folder if not exist
# output_directory="results"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
# 
# 
# 
# 
# 
# 
# 
# 
# 4: Decode the raw prediction `y_pred`
 
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
 
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
# data = np.array(y_pred_decoded)
print(y_pred_decoded[0])



      
det_label = 0
det_conf = 0
det_xmin = 0
det_ymin =0
det_xmax = 0
det_ymax = 0
det_conf_max=0
idx_Box=0
for i ,a in enumerate(y_pred_decoded):
    print("box")
#     print(a)
##    print(a[i][1])
    for k,b in enumerate(a):
##        print("conf")
##        print (k,b[1])
        det_conf=b[1]
        
        if det_conf>0.4 and det_conf>det_conf_max:
            idx_Box=k
            det_conf_max=det_conf
##            print (k,b[1])
#             print("MAX=")
#             print(det_conf_max)
# 
#             
            det_label = b[0]
#             print(det_label)
#             
            det_conf = det_conf
#             print(det_conf)
#             
            det_xmin = b[2]
#             print(det_xmin)
#             
            det_ymin = b[3]
#             print(det_ymin)
#             
            det_xmax = b[4]
#             print(det_xmax)
#             
            det_ymax = b[5]
#             print(det_ymax)


            
# 5: Draw the predicted boxes onto the image
# height = batch_images[i].shape[0]
# width = batch_images[i].shape[1]
# batch_images1=np.array(processed_images)
height = defectImage.shape[0]
width = defectImage.shape[1]
 

 
 
img=defectImage
 
 
# Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() 
# Just so we can print class names onto the image instead of IDs
 
 
classes = ['A001', 'B002', 'C003', 'D004', 'E005', 'F006'] 
 
 
 
 
 
# Draw the ground truth boxes in green (omit the label for more clarity)
 
xmin = int(det_xmin)
ymin = int(det_ymin)
xmax = int(det_xmax)
ymax = int(det_ymax)
 
 
print(xmin)
label = '{}'.format(classes[int(det_label)])
img=cv2.rectangle(defectImage,(xmin,ymin),(xmax,ymax),(0,255,2),2)
s = label
img=cv2.putText(img,s,(xmax+2,ymin-2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1)
cv2.moveWindow("Detection",1000,100)
cv2.imshow('Detection',img)
cv2.waitKey(0)  
cv2.destroyAllWindows()  
