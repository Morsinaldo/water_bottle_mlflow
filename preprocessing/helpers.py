import cv2
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array

class SimplePreprocessor:
    """ Simple Preprocessor class to resize the image to a fixed size ignoring the aspect ratio. """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
        return cv2.resize(image, (self.width, self.height),interpolation=self.inter)

class ImageToArrayPreprocessor:
    """ ImageToArrayPreprocessor class to convert the image to a numpy array. """
    def __init__(self, dataFormat=None):
		# store the image data format
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)

class SimpleDatasetLoader:
    """ SimpleDatasetLoader class to load the image dataset from disk and apply the preprocessors. """
    def __init__(self, preprocessors=None, logger=None):
		# store the image preprocessor
        self.preprocessors = preprocessors
        self.logger = logger

		# if the preprocessors are None, initialize them as an
		# empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
            # initialize the list of features and labels
        data = []
        labels = []

            # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            # e.g "img example: ./artifacts/animals_raw_data:v0/dogs/dogs_00892.jpg"
            # imagePath.split(os.path.sep)[-2] will return "dogs"
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
    
            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                self.logger.info("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))

            # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

