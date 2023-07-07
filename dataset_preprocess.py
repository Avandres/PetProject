import os
import sys
import random

import cv2
import numpy as np
from tqdm import trange


class DatasetPreprocessor:

    def __init__(self):
        pass

    def preprocess_image(self, img: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        Preprocesses the image. The preprocessing includes translation to grayscale, image resizing,
        and image normalization (the values of each element in the array are represented in the range from 0 to 1)

        Args:
            img (np.ndarray): An image represented as a numpy-array.
            height (int): The height of the image after preprocessing. Is equal to the number of rows
                          in the resulting array.
            width (int): The width of the image after preprocessing. Is equal to the number of columns
                         in the resulting array.

        Returns:
            normalized_img (np.ndarray): Preprocessed image, represented as a numpy-array
        """
        if height < 1:
            raise ValueError("Argument 'height' must be greater than zero.")
        if width < 1:
            raise ValueError("Argument 'width' must be greater than zero.")
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(grayscale_img, (width, height))
        normalized_img = resized_img / 255.0
        return normalized_img

    def __load_preprocessed_audio(self, filepath: str, height: int, width: int) -> np.ndarray:
        """
        Image loading and preprocessing

        Args:
            filepath (str): The path to the image to be uploaded
            height (int): The height of the image after preprocessing. Is equal to the number of rows
                          in the resulting array.
            width (int): The width of the image after preprocessing. Is equal to the number of columns
                         in the resulting array.

        Returns:
            preprocessed_img (np.ndarray): Preprocessed image, represented as a numpy-array
        """
        img = cv2.imread(filepath)
        preprocessed_img = self.preprocess_image(img, height, width)
        return preprocessed_img

    def __get_same_img_arrays(self, pathslist: list,
                              img_height: int, img_width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Uploading two random different images of the same object

        Args:
            pathslist (list): List of paths to folders corresponding to different objects
            img_height: The height of the image after preprocessing. Is equal to the number of rows
                        in the resulting array.
            img_width: The width of the image after preprocessing. Is equal to the number of columns
                       in the resulting array.

        Returns:
            img1 (np.ndarray): Preprocessed image, represented as a numpy-array
            img2 (np.ndarray): Preprocessed image, represented as a numpy-array
        """
        directories = [directory for directory in pathslist if os.path.isdir(directory)]
        if not directories:
            raise NotADirectoryError("There are no directories on the specified path")
        directory = random.choice(directories)
        imagepaths = [filename for filename in os.listdir(directory)
                      if filename.endswith(".png") or filename.endswith(".jpg")]
        if not imagepaths:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory)
        filename_1, filename_2 = np.random.choice(imagepaths, 2, replace=False)
        imagepath_1 = os.path.join(directory, filename_1)
        imagepath_2 = os.path.join(directory, filename_2)
        img1 = self.__load_preprocessed_audio(imagepath_1, img_height, img_width)
        img2 = self.__load_preprocessed_audio(imagepath_2, img_height, img_width)
        return img1, img2

    def __get_different_img_arrays(self, pathslist: list,
                                   img_height: int, img_width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Uploading images of two different randomly selected objects

        Args:
            pathslist (list): List of paths to folders corresponding to different objects
            img_height: The height of the image after preprocessing. Is equal to the number of rows
                        in the resulting array.
            img_width: The width of the image after preprocessing. Is equal to the number of columns
                       in the resulting array.

        Returns:
            img1 (np.ndarray): Preprocessed image, represented as a numpy-array
            img2 (np.ndarray): Preprocessed image, represented as a numpy-array
        """
        directories = [directory for directory in pathslist if os.path.isdir(directory)]
        if not directories:
            raise NotADirectoryError("There are no directories on the specified path")
        directory_1, directory_2 = np.random.choice(directories, 2, replace=False)
        imagepaths_1 = [filename for filename in os.listdir(directory_1)
                        if filename.endswith(".png") or filename.endswith(".jpg")]
        if not imagepaths_1:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory_1)
        imagepaths_2 = [filename for filename in os.listdir(directory_2)
                        if filename.endswith(".png") or filename.endswith(".jpg")]
        if not imagepaths_2:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory_2)
        imagepath_1 = os.path.join(directory_1, random.choice(imagepaths_1))
        imagepath_2 = os.path.join(directory_2, random.choice(imagepaths_2))
        img1 = self.__load_preprocessed_audio(imagepath_1, img_height, img_width)
        img2 = self.__load_preprocessed_audio(imagepath_2, img_height, img_width)
        return img1, img2

    def get_data(self, sample_size: int, img_height: int,
                 img_width: int, path: str = './/') -> tuple[np.ndarray, np.ndarray]:
        """
        Data generation, which is performed by selecting two randomly selected images of one object,
        as well as images of two randomly selected objects. The main folder pointed to by the path
        argument must contain folders for each unique object. Each of these folders should contain
        images of the corresponding object.

        Args:
            sample_size (int): Number of generated image pairs
            img_height (int): The height of the images in the data sample in pixels. Equal to the number of rows
            img_width (int): The width of the images in the data sample in pixels. Equal to the number of columns
            path (str): The path to the folder that contains the data. There should be other folders in
                        the data folder — one folder for each unique object. Each folder of the object
                        must contain images with this object

        Returns:
            x_data (np.ndarray): Numpy-array containing pairs of images intended for training
                                 the siamese neural network
            y_data (np.ndarray): Numpy-array that contains labels indicating whether the images
                                 contain the same object in the corresponding pair. If the label value
                                 is 1, then the same object is in the pair on both images, if the label
                                 value is 0, then there are different objects on the images
        """
        if sample_size < 1:
            raise ValueError("Argument 'sample_size' must be greater than zero.")
        if img_height < 1:
            raise ValueError("Argument 'img_height' must be greater than zero.")
        if img_width < 1:
            raise ValueError("Argument 'img_width' must be greater than zero.")
        x_data = np.zeros([sample_size, 2, img_height, img_width, 1])
        y_data = np.zeros([sample_size, 1])
        pathslist = [os.path.join(path, filepath) for filepath in os.listdir(path)]
        print("Формирование выборки данных: ")
        for i in trange(sample_size, file=sys.stdout, colour='white'):
            if i % 2 == 0:
                img1_array, img2_array = self.__get_same_img_arrays(pathslist, img_height, img_width)
                y_data[i] = 1
            else:
                img1_array, img2_array = self.__get_different_img_arrays(pathslist, img_height, img_width)
                y_data[i] = 0
            x_data[i, 0, :, :, 0] = img1_array
            x_data[i, 1, :, :, 0] = img2_array
        return x_data, y_data
