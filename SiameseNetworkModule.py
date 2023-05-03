import os
import random

import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(y_true, y_pred):
    return y_true[y_pred.ravel() < 0.5].mean()

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# Тест!!! Убрать!!!
def preprocess_image(img, height, width):
    grayscale_img = img.convert('L')
    resized_img = grayscale_img.resize((height, width))
    float_img_array = np.asarray(resized_img).astype('float32')
    normalized_img_array = float_img_array / 255.0
    return normalized_img_array

def load_preprocessed_image(filepath, height, width):
    img = Image.open(filepath)
    img_array = preprocess_image(img, height, width)
    return img_array
# Тест!!! Убрать!!!


class SiameseNetworkClass:

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.__build_model(self.input_dim)
        self.model.compile(loss=contrastive_loss,
                           optimizer=RMSprop(0.0005),
                           metrics=[accuracy]
                           )
        self.__EPOCHS = 100
        self.__BATCH_SIZE = 128

    def __build_model(self, input_dim):
        input_a = Input(shape=input_dim)
        input_b = Input(shape=input_dim)
        base_network = self.__build_base_network(input_dim)
        feature_vector_a = base_network(input_a)
        feature_vector_b = base_network(input_b)
        distance = Lambda(self.__euclidean_distance,
                          output_shape=self.__dist_output_shape)([feature_vector_a, feature_vector_b])
        return Model(inputs=[input_a, input_b], outputs=distance)

    def __euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        euclidean_distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
        return euclidean_distance

    def __dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def __build_base_network(self, input_shape):
        seq = Sequential()
        # convolutional layer 1
        seq.add(Convolution2D(6, (3, 3), input_shape=input_shape,
                              padding='valid', data_format="channels_last", activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))
        # convolutional layer 2
        seq.add(Convolution2D(12, (3, 3), padding='valid', data_format="channels_last", activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))
        # flatten
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(50, activation='relu'))
        return seq

    def load_weights(self, path_to_weights="default_weights.ckpt"):
        self.model.load_weights(path_to_weights)

    def save_weights(self, path_to_weights='users_weights.ckpt'):
        self.model.save_weights(path_to_weights)

    def train_model(self, x_train, y_train, x_test=None, y_test=None):
        images_for_input_a = x_train[:, 0]
        images_for_input_b = x_train[:, 1]
        self.model.fit([images_for_input_a, images_for_input_b],
                       y_train,
                       batch_size=128,
                       verbose=1,
                       epochs=self.__EPOCHS
                       )
        if not (x_test is None or y_test is None):
            test_prediction = self.model.predict(x_test)
            test_accuracy = accuracy(y_test, test_prediction)
            print("Test accuracy: ", test_accuracy)

    def make_prediction(self, image1, image2):
        image1 = image1[None, ..., None]
        image2 = image2[None, ..., None]
        return self.model.predict([image1, image2])


if __name__ == '__main__':
    my_network = SiameseNetworkClass((50, 50, 1))
    my_network.load_weights('users_weights.ckpt')
    photo1 = r'C:\Work\Сиамские_сети\signatures\s162\1.png'
    photo2 = r'C:\Work\Сиамские_сети\signatures\s162\2.png'
    img1 = load_preprocessed_image(photo1, 50, 50)
    img2 = load_preprocessed_image(photo2, 50, 50)
    print(my_network.make_prediction(img1, img2))
