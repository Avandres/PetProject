import os
import random

import numpy as np
from PIL import Image
from tqdm import trange
import sys


class DatasetPreprocessor:

    def __init__(self):
        pass

    def preprocess_image(self, img, height, width):
        grayscale_img = img.convert('L')
        resized_img = grayscale_img.resize((width, height))
        float_img_array = np.asarray(resized_img).astype('float32')
        normalized_img_array = float_img_array / 255.0
        return normalized_img_array

    def __load_preprocessed_image(self, filepath, height, width):
        img = Image.open(filepath)
        img_array = self.preprocess_image(img, height, width)
        return img_array

    def __get_same_img_arrays(self, directories, img_height, img_width, path=".//"):
        directory = random.choice(directories)
        filepaths = os.listdir(path + '//' + directory)
        filename_1, filename_2 = np.random.choice(filepaths, 2, replace=False)
        filepath_1 = path + '//' + directory + '//' + filename_1
        filepath_2 = path + '//' + directory + '//' + filename_2
        img1_array = self.__load_preprocessed_image(filepath_1, img_height, img_width)
        img2_array = self.__load_preprocessed_image(filepath_2, img_height, img_width)
        return img1_array, img2_array

    def __get_different_img_arrays(self, directories, img_height, img_width, path=".//"):
        directory_1, directory_2 = np.random.choice(directories, 2, replace=False)
        filepath_1 = path + '//' + directory_1 + '//' + random.choice(os.listdir(path + '//' + directory_1))
        filepath_2 = path + '//' + directory_2 + '//' + random.choice(os.listdir(path + '//' + directory_2))
        img1_array = self.__load_preprocessed_image(filepath_1, img_height, img_width)
        img2_array = self.__load_preprocessed_image(filepath_2, img_height, img_width)
        return img1_array, img2_array

    def get_data(self, total_sample_size, img_height, img_width, path='.//'):
        if total_sample_size < 1:
            raise ValueError("Argument 'total_sample_size' must be greater than zero.")
        X = np.zeros([total_sample_size, 2, img_height, img_width, 1])
        Y = np.zeros([total_sample_size, 1])
        directories = os.listdir(path)
        print("Формирование выборки данных: ")
        for i in trange(total_sample_size, file=sys.stdout, colour='white'):
            if i % 2 == 0:
                img1_array, img2_array = self.__get_same_img_arrays(directories, img_height, img_width, path=path)
                Y[i] = 1
            else:
                img1_array, img2_array = self.__get_different_img_arrays(directories, img_height, img_width, path=path)
                Y[i] = 0
            X[i, 0, :, :, 0] = img1_array
            X[i, 1, :, :, 0] = img2_array
        return X, Y

if __name__ == '__main__':
    total_sample_size = 5000
    path = 'C://Work//PetProject//archive//Extracted Faces'
    x_data, y_data = DatasetPreprocessor().get_data(total_sample_size, 50, 50, path=path)
    print(x_data)