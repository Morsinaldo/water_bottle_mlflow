"""
Creator: Morsinaldo Medeiros
Date: 06-02-2023
Description: This script contains the preprocessing classes.
"""

import os
import cv2
import numpy as np

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
