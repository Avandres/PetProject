from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Activation


def contrastive_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """
    Calculates the contrastive loss for the siamese neural network. More detailed:
    https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.html

    Args:
        y_true (np.ndarray): A vector of true values indicating whether one object is located in two photos,
                             or different objects. The values can be either 0 or 1.
        y_pred (np.ndarray): Vector of predicted values

    Returns:
        loss (Tensor): Calculated value of the contrastive loss function
    """
    margin = 1
    loss = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return loss


def compute_test_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """
    Calculates the accuracy of Siamese neural network predictions during training.
    The forecast is considered correct if y_pred < 0.5

    Args:
        y_true (np.ndarray): A vector of true values indicating whether one object is located in two photos,
                             or different objects. The values can be either 0 or 1.
        y_pred (np.ndarray): Vector of predicted values

    Returns:
        Tensor: Accuracy for Siamese neural network predictions
    """
    return y_true[y_pred.ravel() < 0.5].mean()


def compute_train_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """
    Calculates the accuracy of Siamese neural network predictions during test.
    The forecast is considered correct if y_pred < 0.5

    Args:
        y_true (np.ndarray): A vector of true values indicating whether one object is located in two photos,
                             or different objects. The values can be either 0 or 1.
        y_pred (np.ndarray): Vector of predicted values

    Returns:
        Tensor: Accuracy for Siamese neural network predictions
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class SiameseConv2d:
    """
    A class for representing a Siamese neural network with convolutional layers.
    More about Siamese neural networks: http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf

    Args:
        input_shape (tuple): The shape of the inputs of the Siamese neural network. It must be equal to the
                             shape of the images that will be fed into the model.

    Methods:
        load_weights: Loads the values of the previously saved weights of the model

        save_weights: Saves the values of the model weights

        fit_model: Training of the Siamese neural network model

        make_prediction: Prediction of the distance between vector-features of two images. The smaller the distance,
                         the more likely it is that the same object is present in two different images

    """

    def __init__(self, input_shape: tuple):
        """
        Sets all the necessary attributes for the SiameseConv2d object

        Args:
            input_shape (tuple): The shape of the inputs of the Siamese neural network. It must be equal to the
                                 shape of the images that will be fed into the model.
        """
        self.input_shape = input_shape
        self.model = self.__build_model()
        self.model.compile(loss=contrastive_loss,
                           optimizer=RMSprop(0.0005),
                           metrics=[compute_train_accuracy]
                           )
        self.__BATCH_SIZE = 128

    def __build_model(self) -> Model:
        """
        Builds a model of the Siamese neural network. To determine the dimension of the inputs,
        self.input_shape is used, specified in the class constructor

        Returns:
            Model: A Siamese neural network model with two inputs and one output. Each input can take a photo,
                   and the distance between the vector-features of two photos is calculated at the output
        """
        first_image_input = Input(shape=self.input_shape)
        second_image_input = Input(shape=self.input_shape)
        base_network = self.__build_base_network()
        first_feature_vector = base_network(first_image_input)
        second_feature_vector = base_network(second_image_input)
        distance = Lambda(self.__euclidean_distance,
                          output_shape=self.__distance_output_shape)([first_feature_vector, second_feature_vector])
        return Model(inputs=[first_image_input, second_image_input], outputs=distance)

    def __euclidean_distance(self, vectors: list[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Calculating the Euclidean distance between two vectors

        Args:
            vectors (list[Tensor, Tensor]): Two vectors (tensors) between which it is necessary
                                            to calculate the euclidean distance

        Returns:
            euclidean_distance (Tensor): A vector with one element that is equal
                                         to the Euclidean distance between two vectors
        """
        sum_square = K.sum(K.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
        euclidean_distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
        return euclidean_distance

    def __distance_output_shape(self, shapes: tuple) -> tuple:
        """
        Calculates the shape of the euclidean distance vector

        Args:
            shapes (tuple): A tuple with shapes of two vectors between which the euclidean distance is calculated

        Returns:
            tuple: The shape of the vector in which the euclidean distance is located
        """
        shape1, shape2 = shapes
        return shape1[0], 1

    def __build_base_network(self) -> Sequential:
        """
        Building a basic neural network model

        Returns:
            Sequential: the basic model common to the two inputs of the siamese neural network
        """
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
        """
        Loads the values of the previously saved weights of the model
        
        Args:
            path_to_weights (str): The path where the weights of the model will be loaded

        Returns:
            None
        """
        self.model.load_weights(path_to_weights)

    def save_weights(self, path_to_weights: str = 'users_weights.ckpt'):
        """
        Saves the values of the model weights
        
        Args:
            path_to_weights (str): The path by which the weights of the model will be saved

        Returns:
            None
        """
        self.model.save_weights(path_to_weights)

    def fit_model(self, x_train: np.ndarray, y_train: np.ndarray, validation_data: Union[tuple, list, None] = None,
                  verbose: int = 1, epochs: int = 100) -> tf.keras.callbacks.History:
        """
        Training of the Siamese neural network model

        Args:
            x_train (np.ndarray): An array of dimensions (sample_size, 2, img_height, img_width, number_of_channels).
                                  Contains pairs of images
            y_train (np.ndarray): A vector of values indicating whether one object is located in two images,
                                  or different objects. The values can be either 0 or 1.
            validation_data (tuple | list | None): Data for model validation. The first element of validation_data is
                                                   x_test and the second element is y_test. If None is supplied as
                                                   validation_data, then the model is not tested after training.
            verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
                           Identical to the model parameter of the same name from tensorflow
            epochs (int): The number of epochs during which the model is trained

        Returns:
            History: The history of model training. Identical to the history of the fit method in keras
        """
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
            test_accuracy = compute_test_accuracy(y_test, test_prediction)
            print("Test accuracy: ", test_accuracy)
        return history

    def make_prediction(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Prediction of the distance between vector-features of two images. The smaller the distance,
        the more likely it is that the same object is present in two different images

        Args:
            image1: Numpy is an array corresponding to the first image.
                    Has dimension (img_height, img_width, number_of_channels)
            image2: Numpy is an array corresponding to the second image.
                    Has dimension (img_height, img_width, number_of_channels)

        Returns:
            float: The value of the distance between the vector features of two images
        """
        if image1.shape != self.input_shape:
            raise ValueError("The shape of 'image1' and the shape "
                             "specified in the constructor of SiameseConv2d don't match")
        if image2.shape != self.input_shape:
            raise ValueError("The shape of 'image2' and the shape "
                             "specified in the constructor of SiameseConv2d don't match")
        image1 = image1[None, ...]
        image2 = image2[None, ...]
        return self.model.predict([image1, image2], verbose=0)[0, 0]
