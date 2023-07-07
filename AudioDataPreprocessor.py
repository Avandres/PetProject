import os
import sys
import random

import librosa
import numpy as np
from tqdm import trange


class AudioPreprocessor:

    def __init__(self):
        pass

    def __load_preprocessed_audio(self, filepath: str, mfcc_num: int) -> np.ndarray:

        y, sr = librosa.load(filepath)
        mfcc_coefs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_num)
        return mfcc_coefs

    def __get_same_audio_arrays(self, pathslist: list, mfcc_num: int) -> tuple[np.ndarray, np.ndarray]:

        directories = [directory for directory in pathslist if os.path.isdir(directory)]
        if not directories:
            raise NotADirectoryError("There are no directories on the specified path")
        directory = random.choice(directories)
        imagepaths = [filename for filename in os.listdir(directory) if filename.endswith(".wav")]
        if not imagepaths:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory)
        filename_1, filename_2 = np.random.choice(imagepaths, 2, replace=False)
        audiopath_1 = os.path.join(directory, filename_1)
        audiopath_2 = os.path.join(directory, filename_2)
        audio1 = self.__load_preprocessed_audio(audiopath_1, mfcc_num)
        audio2 = self.__load_preprocessed_audio(audiopath_2, mfcc_num)
        return audio1, audio2

    def __get_different_audio_arrays(self, pathslist: list, mfcc_num: int) -> tuple[np.ndarray, np.ndarray]:

        directories = [directory for directory in pathslist if os.path.isdir(directory)]
        if not directories:
            raise NotADirectoryError("There are no directories on the specified path")
        directory_1, directory_2 = np.random.choice(directories, 2, replace=False)
        imagepaths_1 = [filename for filename in os.listdir(directory_1)
                        if filename.endswith(".wav")]
        if not imagepaths_1:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory_1)
        imagepaths_2 = [filename for filename in os.listdir(directory_2)
                        if filename.endswith(".wav")]
        if not imagepaths_2:
            raise FileNotFoundError("There are no *.png or *.jpg files on the "
                                    + directory_2)
        audiopath_1 = os.path.join(directory_1, random.choice(imagepaths_1))
        audiopath_2 = os.path.join(directory_2, random.choice(imagepaths_2))
        audio1 = self.__load_preprocessed_audio(audiopath_1, mfcc_num)
        audio2 = self.__load_preprocessed_audio(audiopath_2, mfcc_num)
        return audio1, audio2

    def get_max_length(self, path: str = './/'):
        max_length = 0
        directories = os.listdir(path)
        for directory in directories:
            for i in range(21):
                try:
                    y, sr = librosa.load(
                        path + '/' + directory + "/" + str(i) + ".wav")
                    mfcc_arr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    if max_length < mfcc_arr.shape[1]:
                        max_length = mfcc_arr.shape[1]
                except:
                    continue
        return max_length

    def get_data(self, sample_size: int, mfcc_num: int, path: str = './/') -> tuple[np.ndarray, np.ndarray]:

        if sample_size < 1:
            raise ValueError("Argument 'sample_size' must be greater than zero.")
        if mfcc_num < 1:
            raise ValueError("Argument 'mfcc_num' must be greater than zero.")
        pad_length = self.get_max_length(path=path)
        x_data = np.zeros([sample_size, 2, mfcc_num, pad_length])
        y_data = np.zeros([sample_size, 1])
        pathslist = [os.path.join(path, filepath) for filepath in os.listdir(path)]
        print("Формирование выборки данных: ")
        for i in trange(sample_size, file=sys.stdout, colour='white'):
            if i % 2 == 0:
                audio1_array, audio2_array = self.__get_same_audio_arrays(pathslist, mfcc_num)
                y_data[i] = 1
            else:
                audio1_array, audio2_array = self.__get_different_audio_arrays(pathslist, mfcc_num)
                y_data[i] = 0
            x_data[i, 0, :, :audio1_array.shape[1]] = audio1_array
            x_data[i, 1, :, :audio2_array.shape[1]] = audio2_array
        #x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2]*x_data.shape[3]))
        return x_data, y_data