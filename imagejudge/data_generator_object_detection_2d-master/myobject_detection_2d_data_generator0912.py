'''
A data generator for 2D object detection.

Copyright (C) 2018 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division
import numpy as np
import cv2
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
from tqdm import tqdm
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from transformations.object_detection_2d_image_boxes_validation_utils import BoxFilter

class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data: A general-purpose CSV parser,
    an XML parser for the Pascal VOC datasets, and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 imageFullPath=""):
        '''
        This class provides parser methods that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the dataset from CSV or XML files. If you already
        have the image filenames and labels in asuitable format (see argument descriptions below), you can pass
        them right here in the constructor, in which case you do not need to call any of the parser methods afterwards.

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file. Defaults to 'text'.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant. Defaults to `None`.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
        '''
#         self.labels_output_format = labels_output_format
#         self.labels_format={'class_id': labels_output_format.index('class_id'),
#                             'xmin': labels_output_format.index('xmin'),
#                             'ymin': labels_output_format.index('ymin'),
#                             'xmax': labels_output_format.index('xmax'),
#                             'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

        # The variables `self.filenames`, `self.labels`, and `self.image_ids` below store the output from the parsers.
        # This is the input for the `generate()`` method. `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves.
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        # Setting `self.labels` is optional, the generator also works if `self.labels` remains `None`.

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
        else:
            self.filenames = []

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

#     def parse_xml(self,
#                   images_dirs,
#                   image_set_filenames,
#                   annotations_dirs=[],
#                   classes=['background',
#                            'aeroplane', 'bicycle', 'bird', 'boat',
#                            'bottle', 'bus', 'car', 'cat',
#                            'chair', 'cow', 'diningtable', 'dog',
#                            'horse', 'motorbike', 'person', 'pottedplant',
#                            'sheep', 'sofa', 'train', 'tvmonitor'],
#                   include_classes = 'all',
#                   exclude_truncated=False,
#                   exclude_difficult=False,
#                   ret=False):
    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=[],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
        # Set class members.
        self.images_dirs = images_dirs
#         self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        print(image_set_filenames)
#         self.classes = classes
#         self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
#         self.labels = []
#         if not annotations_dirs:
#             self.labels = None
#             annotations_dirs = [None] * len(images_dirs)
#             print(annotations_dirs)

#         for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
        for images_dir, image_set_filename in zip(images_dirs, image_set_filenames):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids
                print("self.image_ids")
                print(self.image_ids)

            # Loop over all images in this dataset.
            #for image_id in image_ids:
            for image_id in tqdm(image_ids, desc=os.path.basename(image_set_filename)):

                filename = '{}'.format(image_id) + '.jpg'
                print("filename")
                print(filename)
                self.filenames.append(os.path.join(images_dir, filename))
                print("my image test....")
                print(os.path.join(images_dir, filename))

#                 if not annotations_dir is None:
#                     # Parse the XML file for this image.
#                     with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
#                         soup = BeautifulSoup(f, 'xml')
# 
#                     folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
#                     #filename = soup.filename.text
# 
#                     boxes = [] # We'll store all boxes for this image here
#                     objects = soup.find_all('object') # Get a list of all objects in this image

#                     # Parse the data for each object
#                     for obj in objects:
#                         class_name = obj.find('name').text
#                         class_id = self.classes.index(class_name)
#                         # Check if this class is supposed to be included in the dataset
#                         if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
#                         pose = obj.pose.text
#                         truncated = int(obj.truncated.text)
#                         if exclude_truncated and (truncated == 1): continue
#                         difficult = int(obj.difficult.text)
#                         if exclude_difficult and (difficult == 1): continue
#                         xmin = int(obj.bndbox.xmin.text)
#                         ymin = int(obj.bndbox.ymin.text)
#                         xmax = int(obj.bndbox.xmax.text)
#                         ymax = int(obj.bndbox.ymax.text)
#                         item_dict = {'folder': folder,
#                                      'image_name': filename,
#                                      'image_id': image_id,
#                                      'class_name': class_name,
#                                      'class_id': class_id,
#                                      'pose': pose,
#                                      'truncated': truncated,
#                                      'difficult': difficult,
#                                      'xmin': xmin,
#                                      'ymin': ymin,
#                                      'xmax': xmax,
#                                      'ymax': ymax}
#                         box = []
#                         for item in self.labels_output_format:
#                             box.append(item_dict[item])
#                         boxes.append(box)
# 
#                     self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels, self.image_ids

    def set_image(self,
                 imageFileName=""):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
#         # Set class members.
#         self.images_dirs = images_dirs
# #         self.annotations_dirs = annotations_dirs
#         self.image_set_filenames = image_set_filenames
#         print(image_set_filenames)
# #         self.classes = classes
# #         self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
#         self.image_ids = []
#         self.labels = []
#         if not annotations_dirs:
#             self.labels = None
#             annotations_dirs = [None] * len(images_dirs)
#             print(annotations_dirs)

#         for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
#         for images_dir, image_set_filename in zip(images_dirs, image_set_filenames):
#             # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
#             with open(image_set_filename) as f:
#                 image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
#                 self.image_ids += image_ids
#                 print("self.image_ids")
#                 print(self.image_ids)

            # Loop over all images in this dataset.
            #for image_id in image_ids:
#             for image_id in tqdm(image_ids, desc=os.path.basename(image_set_filename)):
# 
#                 filename = '{}'.format(image_id) + '.jpg'
#                 print("filename")
#                 print(filename)
        self.filenames.append(imageFileName)
        print("my image test....")
        print(imageFileName)

#                 if not annotations_dir is None:
#                     # Parse the XML file for this image.
#                     with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
#                         soup = BeautifulSoup(f, 'xml')
# 
#                     folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
#                     #filename = soup.filename.text
# 
#                     boxes = [] # We'll store all boxes for this image here
#                     objects = soup.find_all('object') # Get a list of all objects in this image

#                     # Parse the data for each object
#                     for obj in objects:
#                         class_name = obj.find('name').text
#                         class_id = self.classes.index(class_name)
#                         # Check if this class is supposed to be included in the dataset
#                         if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
#                         pose = obj.pose.text
#                         truncated = int(obj.truncated.text)
#                         if exclude_truncated and (truncated == 1): continue
#                         difficult = int(obj.difficult.text)
#                         if exclude_difficult and (difficult == 1): continue
#                         xmin = int(obj.bndbox.xmin.text)
#                         ymin = int(obj.bndbox.ymin.text)
#                         xmax = int(obj.bndbox.xmax.text)
#                         ymax = int(obj.bndbox.ymax.text)
#                         item_dict = {'folder': folder,
#                                      'image_name': filename,
#                                      'image_id': image_id,
#                                      'class_name': class_name,
#                                      'class_id': class_id,
#                                      'pose': pose,
#                                      'truncated': truncated,
#                                      'difficult': difficult,
#                                      'xmin': xmin,
#                                      'ymin': ymin,
#                                      'xmax': xmax,
#                                      'ymax': ymax}
#                         box = []
#                         for item in self.labels_output_format:
#                             box.append(item_dict[item])
#                         boxes.append(box)
# 
#                     self.labels.append(boxes)

#         if ret:
#             return self.filenames, self.labels, self.image_ids

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple with the processed images as its first element and, if labels and a label encoder are given,
                the encoded labels as its second element. Apart from that, the output tuple can contain additional outputs
                according to the keywords specified here. The possible keyword strings and their respective outputs are:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                    but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                    in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument. By default, this will be
            a 2-tuple containing the processed batch images as its first element and the encoded ground truth boxes
            tensor as its second element if in training mode, or a 1-tuple containing only the processed batch images if
            not in training mode. Any additional outputs must be specified in the `returns` argument.
        '''

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

#         if self.labels is None:
#             if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors']]):
#                 warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'encoded_labels', and 'matched_anchors' " +
#                               "are possible returns, but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))
#         elif label_encoder is None:
#             if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
#                 warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
#                               "but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))
#         elif not isinstance(label_encoder, SSDInputEncoder):
#             if 'matched_anchors' in returns:
#                 warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
#                               "but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))
#         if (self.image_ids is None) and ('image_ids' in returns):
#             warnings.warn("No image IDs were given, therefore 'image_ids' is not a possible return, " +
#                           "but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

#         if shuffle:
#             if (self.labels is None) and (self.image_ids is None):
#                 self.filenames = sklearn.utils.shuffle(self.filenames)
#             elif (self.labels is None):
#                 self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
#             elif (self.image_ids is None):
#                 self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
#             else:
#                 self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)

#         if degenerate_box_handling == 'remove':
#             box_filter = BoxFilter(check_overlap=False,
#                                    check_min_area=False,
#                                    check_degenerate=True,
#                                    labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
#         if not (self.labels is None):
#             for transform in transformations:
#                 transform.labels_format = self.labels_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0
        k=0

        while True:

            batch_X= []

            if current >= len(self.filenames):
                current = 0

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

#                 if shuffle:
#                     if (self.labels is None) and (self.image_ids is None):
#                         self.filenames = sklearn.utils.shuffle(self.filenames)
#                     elif (self.labels is None):
#                         self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
#                     elif (self.image_ids is None):
#                         self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
#                     else:
#                         self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)

            #########################################################################################
            # Get the images, image filenames, (maybe) image IDs, and (maybe) labels for this batch.
            #########################################################################################

            # Get the image filepaths for this batch.
            batch_filenames = self.filenames[current:current+batch_size]

            # Load the images for this batch.
            for filename in batch_filenames:
#                 with Image.open(filename) as img:
                img=cv2.imread(filename)    
                print(filename)
                batch_X.append(np.array(img))
                print(" batch_X.append(np.array(img))")
                print(batch_X)
                cv2.imshow("read123", np.array(img,dtype=np.uint8))
                cv2.imwrite("D:/2.jpg",np.array(img))
                cv2.waitKey(0)
                cv2.destroyAllWindows()   
             
                print(" batch_X.append(np.array(img))")

#             # Get the labels for this batch (if there are any).
#             if not (self.labels is None):
#                 batch_y = deepcopy(self.labels[current:current+batch_size])

            # Get the image IDs for this batch (if there are any).
#             if not (self.image_ids is None):
#                 batch_image_ids = self.image_ids[current:current+batch_size]

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
#             if 'original_labels' in returns and not self.labels is None:
#                 batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

#                 if not (self.labels is None):
#                     # Convert the labels for this image to an array (in case they aren't already).
#                     batch_y[i] = np.array(batch_y[i])
#                     # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
#                     if (batch_y[i].size == 0) and not keep_images_without_gt:
#                         batch_items_to_remove.append(i)
#                         batch_inverse_transforms.append([])
#                         continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
#                                 batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
#                                 batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])
                                batch_X[i] = transform(batch_X[i])
                                

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])
              
#                 #########################################################################################
#                 # Check for degenerate boxes in this batch item.
#                 #########################################################################################
# 
#                 xmin = self.labels_format['xmin']
#                 ymin = self.labels_format['ymin']
#                 xmax = self.labels_format['xmax']
#                 ymax = self.labels_format['ymax']
# 
# 
# 
# 
#   
#            
# 
#                 # Check for degenerate ground truth bounding boxes before attempting any computations.
#                 if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
#                     if degenerate_box_handling == 'warn':
#                         warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
#                                       "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
#                                       "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
#                                       "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
#                                       "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
#                     elif degenerate_box_handling == 'remove':
#                         batch_y[i] = box_filter(batch_y[i])
#                         if (batch_y[i].size == 0) and not keep_images_without_gt:
#                             batch_items_to_remove.append(i)
# 
#             #########################################################################################
#             # Remove any items we might not want to keep from the batch.
#             #########################################################################################
# 
#             if batch_items_to_remove:
#                 for j in sorted(batch_items_to_remove, reverse=True):
#                     # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
#                     batch_X.pop(j)
#                     batch_filenames.pop(j)
#                     if batch_inverse_transforms: batch_inverse_transforms.pop(j)
#                     if not (self.labels is None): batch_y.pop(j)
#                     if not (self.image_ids is None): batch_image_ids.pop(j)
#                     if 'original_images' in returns: batch_original_images.pop(j)
#                     if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)
# 
#             #########################################################################################
# 
#             # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
#             #          or varying numbers of channels. At this point, all images must have the same size and the same
#             #          number of channels.
#             batch_X = np.array(batch_X)
#             if (batch_X.size == 0):
#                 raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
#                                            "in their size and/or number of channels. Note that after all transformations " +
#                                            "(if any were given) have been applied to all images in the batch, all images " +
#                                            "must be homogenous in size along all axes.")
# 
#             #########################################################################################
#             # If we have a label encoder, encode our labels.
#             #########################################################################################
# 
#             if not (label_encoder is None or self.labels is None):
#                 if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
#                     batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
#                 else:
#                     batch_y_encoded = label_encoder(batch_y, diagnostics=False)
# 
#             #########################################################################################
#             # Compose the output.
#             #########################################################################################

            ret = []
#             ret.append(batch_X)
            ret.append(np.array(batch_X))
#             print("test12")
#             print(len(batch_X))
#             print(type(batch_X))
#             print(batch_X)
#             print("test1")
#             cv2.imshow("win", np.array(batch_X[0]))
#             cv2.imwrite("D:/temp/cv/ssd_keras-master/datasets/udacity_driving_datasets/1.jpg",np.array(batch_X[k]))
#             k=k+1
#            
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()   
            
            
            print("ret12")
            print(len(ret))
            print(type(ret))
            print(ret[0])
            print("ret13")
            
#             testImg=ret(0)
#             testImg=np.array(testImg[k])
            cv2.imshow("ret1", ret[0])
            cv2.imwrite("D:/temp/cv/ssd_keras-master/datasets/udacity_driving_datasets/2.jpg",np.array(batch_X[k]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()   
             
            
          
            
            
#             if not (label_encoder is None or self.labels is None):
#                 ret.append(batch_y_encoded)
#                 if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder): ret.append(batch_matched_anchors)
#             if 'processed_labels' in returns and not self.labels is None: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
#             if 'image_ids' in returns and not self.image_ids is None: ret.append(batch_image_ids)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
#             if 'original_labels' in returns and not self.labels is None: ret.append(batch_original_labels)
            yield ret
#             return ret
    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None):
        '''
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)

    def get_dataset(self):
        '''
        Returns:
            The list of filenames, the list of labels, and the list of image IDs.
        '''
        return self.filenames, self.labels, self.image_ids

    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return len(self.filenames)
