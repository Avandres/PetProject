import os

import librosa
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

import AudioDataPreprocessor

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

def euclidean_distance(vectors: list[tf.Tensor, tf.Tensor]) -> tf.Tensor:
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

def distance_output_shape(shapes: tuple) -> tuple:
    """
    Calculates the shape of the euclidean distance vector

    Args:
        shapes (tuple): A tuple with shapes of two vectors between which the euclidean distance is calculated

    Returns:
        tuple: The shape of the vector in which the euclidean distance is located
    """
    shape1, shape2 = shapes
    return shape1[0], 1



x_data, y_data = AudioDataPreprocessor.AudioPreprocessor().get_data(sample_size=8000,
                                                                    mfcc_num=20,
                                                                    path=r'C:\Work\PetProject\archive\Audio Commands')

first_input = layers.Input(shape=x_data[0, 0].shape)
second_input = layers.Input(shape=x_data[0, 0].shape)

base_network = tf.keras.Sequential()
base_network.add(layers.Bidirectional(layers.LSTM(64,
                                      #return_sequences=False
                                      )))
base_network.add(layers.Dense(10, activation="relu"))

first_feature_vector = base_network(first_input)
second_feature_vector = base_network(second_input)
distance = layers.Lambda(euclidean_distance,
                  output_shape=distance_output_shape)([first_feature_vector, second_feature_vector])
model = tf.keras.Model(inputs=[first_input, second_input], outputs=distance)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

first_train_audio, second_train_audio = x_train[:, 0], x_train[:, 1]
first_test_audio, second_test_audio = x_test[:, 0], x_test[:, 1]

model.compile(loss=contrastive_loss,
              optimizer=RMSprop(0.0005),
              metrics=[compute_train_accuracy]
              )

history = model.fit([first_train_audio, second_train_audio],
                     y_train,
                     batch_size=128,
                     verbose=1,
                     epochs=100,
                     validation_data=[[first_test_audio, second_test_audio], y_test]
                     )

import math
import struct
from collections import deque

import pyaudio
import wave
import librosa

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECOND = 1
THRESHOLD = 0.03

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

start_stop_flag = False

def get_rms(data):
    count = len(data) / 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, data)

    sum_squares = 0.0
    for sample in shorts:
        n = sample / 32768.0
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


while True:
    current_x = np.zeros([1, 2, 20, x_data[0, 0, 0].shape[0]])
    frames = []
    last_data = deque(maxlen=20)
    timer = 0
    print('Start...')
    while True:
        data = stream.read(CHUNK)
        last_data.append(data)
        if get_rms(data) > THRESHOLD:
            if not start_stop_flag:
                frames += list(last_data)
            start_stop_flag = True
            timer = 0
        if start_stop_flag:
            if timer >= RATE // CHUNK * RECORD_SECOND:
                break
            timer += 1
            frames.append(data)

    print('...End.')

    wf = wave.open('test_wave.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    y, sr = librosa.load('test_wave.wav')
    mfcc_coefs1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    if mfcc_coefs1.shape[1] > x_data[0, 0, 0].shape[0]:
        current_x[0, 0, :, :x_data[0, 0, 0].shape[0]] = mfcc_coefs1[:, :x_data[0, 0, 0].shape[0]]
    else:
        current_x[0, 0, :, :mfcc_coefs1.shape[1]] = mfcc_coefs1

    input("Нажмите Enter, чтобы продолжить.")
    time.sleep(1)

    frames = []
    last_data = deque(maxlen=20)
    timer = 0
    print('Start...')
    while True:
        data = stream.read(CHUNK)
        last_data.append(data)
        if get_rms(data) > THRESHOLD:
            if not start_stop_flag:
                frames += list(last_data)
            start_stop_flag = True
            timer = 0
        if start_stop_flag:
            if timer >= RATE // CHUNK * RECORD_SECOND:
                break
            timer += 1
            frames.append(data)

    print('...End.')
    wf = wave.open('test_wave.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    y, sr = librosa.load('test_wave.wav')
    mfcc_coefs2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    if mfcc_coefs2.shape[1] > x_data[0, 0, 0].shape[0]:
        current_x[0, 0, :, :x_data[0, 0, 0].shape[0]] = mfcc_coefs2[:, :x_data[0, 0, 0].shape[0]]
    else:
        current_x[0, 0, :, :mfcc_coefs2.shape[1]] = mfcc_coefs2

    first_train_audio, second_train_audio = current_x[:, 0], current_x[:, 1]
    print(model.predict([first_train_audio, second_train_audio]))
    input("Нажмите Enter, чтобы продолжить.")
    time.sleep(1)


stream.stop_stream()
stream.close()
p.terminate()



