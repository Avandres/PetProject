import os
import random
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop


def contrastive_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    return y_true[y_pred.ravel() < 0.5].mean()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class SiameseConv2d:

    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.model = self.__build_model()
        self.model.compile(loss=contrastive_loss,
                           optimizer=RMSprop(0.0005),
                           metrics=[accuracy]
                           )
        self.__BATCH_SIZE = 128

    def __build_model(self) -> Model:
        first_image_input = Input(shape=self.input_shape)
        second_image_input = Input(shape=self.input_shape)
        base_network = self.__build_base_network()
        first_feature_vector = base_network(first_image_input)
        second_feature_vector = base_network(second_image_input)
        distance = Lambda(self.__euclidean_distance,
                          output_shape=self.__distance_output_shape)([first_feature_vector, second_feature_vector])
        return Model(inputs=[first_image_input, second_image_input], outputs=distance)

    def __euclidean_distance(self, vectors: list[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        sum_square = K.sum(K.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
        euclidean_distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
        return euclidean_distance

    def __distance_output_shape(self, shapes: tuple) -> tuple:
        shape1, shape2 = shapes
        return shape1[0], 1

    def __build_base_network(self) -> Sequential:
        base_network = Sequential()
        # convolutional layer 1
        base_network.add(Convolution2D(6, (3, 3), input_shape=self.input_shape,
                                       padding='valid', data_format="channels_last", activation='relu'))
        base_network.add(MaxPooling2D(pool_size=(2, 2)))
        base_network.add(Dropout(0.25))
        # convolutional layer 2
        base_network.add(Convolution2D(12, (3, 3), padding='valid', data_format="channels_last", activation='relu'))
        base_network.add(MaxPooling2D(pool_size=(2, 2)))
        base_network.add(Dropout(0.25))
        # flatten
        base_network.add(Flatten())
        base_network.add(Dense(128, activation='relu'))
        base_network.add(Dropout(0.1))
        base_network.add(Dense(50, activation='relu'))
        return base_network

    def load_weights(self, path_to_weights: str = "default_weights.ckpt"):
        self.model.load_weights(path_to_weights)

    def save_weights(self, path_to_weights: str = 'users_weights.ckpt'):
        self.model.save_weights(path_to_weights)

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray, validation_data: Union[tuple, list, None] = None,
                    verbose: int = 1, epochs: int = 100) -> tf.keras.callbacks.History:
        if not epochs > 0:
            raise ValueError("Argument 'epochs' must be greater than zero")
        if (x_train.ndim != 5) or (x_train.shape[1] != 2) or (x_train.shape[2:] != self.input_shape):
            raise ValueError("Shape of x_train must be (sample_size, 2, img_height, img_width, number_of_channels)")
        if y_train.ndim != 1:
            raise ValueError("Number of dimensions of y_train must be 1")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in x_train and y_train must be the same")
        images_for_first_input = x_train[:, 0]
        images_for_second_input = x_train[:, 1]
        history = self.model.fit([images_for_first_input, images_for_second_input],
                                 y_train,
                                 batch_size=self.__BATCH_SIZE,
                                 verbose=verbose,
                                 epochs=epochs
                                 )
        if not (validation_data is None):
            x_test = validation_data[0]
            y_test = validation_data[1]
            if (x_test.ndim != 5) or (x_test.shape[1] != 2) or (x_test.shape[2:] != self.input_shape):
                raise ValueError("Shape of validation_data[0] must be (sample_size, 2, img_height, img_width, "
                                 "number_of_channels)")
            if y_test.ndim != 1:
                raise ValueError("Number of dimensions of validation_data[1] must be 1")
            if x_test.shape[0] != y_test.shape[0]:
                raise ValueError("Number of samples in validation_data[0] and validation_data[1] must be the same")
            images_for_first_input = x_test[:, 0]
            images_for_second_input = x_test[:, 1]
            test_prediction = self.model.predict([images_for_first_input, images_for_second_input])
            test_accuracy = accuracy(y_test, test_prediction)
            print("Test accuracy: ", test_accuracy)
        return history

    def make_prediction(self, image1: np.ndarray, image2: np.ndarray) -> float:
        if image1.shape != self.input_shape:
            raise ValueError("The shape of 'image1' and the shape "
                             "specified in the constructor of SiameseConv2d don't match")
        if image2.shape != self.input_shape:
            raise ValueError("The shape of 'image2' and the shape "
                             "specified in the constructor of SiameseConv2d don't match")
        image1 = image1[None, ...]
        image2 = image2[None, ...]
        return self.model.predict([image1, image2], verbose=0)[0, 0]
