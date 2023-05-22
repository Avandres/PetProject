from unittest import TestCase, main
from siamese_networks import SiameseConv2d
import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess_image(img, height, width):
    grayscale_img = img.convert('L')
    resized_img = grayscale_img.resize((height, width))
    float_img_array = np.asarray(resized_img).astype('float32')
    normalized_img_array = float_img_array / 255.0
    return normalized_img_array[..., None]


class SiameseConv2dTest(TestCase):

    def test_train_model_arguments_shape(self):
        x_train = np.array([
            [[[[1.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]],
            [[[[2.0]] * 10 for _ in range(10)], [[[2.0]] * 10 for _ in range(10)]],
            [[[[3.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]]
        ])
        y_train = np.array([1.0, 1.0, 2.0])
        siamese_network = SiameseConv2d((10, 10, 1))
        with self.assertRaises(ValueError) as x_train_exception:
            siamese_network.train_model(np.array([1, 2, 3, 4]), y_train, epochs=1)
            siamese_network.train_model(np.array([[1, 2], [3, 4]]), y_train, epochs=1)
            siamese_network.train_model(np.array([[[1, 2, 3]], [[4, 5, 6]]]), y_train, epochs=1)
        self.assertEqual("Shape of x_train must be (sample_size, 2, img_height, img_width, number_of_channels)",
                         x_train_exception.exception.args[0])
        with self.assertRaises(ValueError) as y_train_exception:
            siamese_network.train_model(x_train, np.array([[1, 2], [3, 4]]), epochs=1)
            siamese_network.train_model(x_train, np.array([[1], [1]]), epochs=1)
        self.assertEqual("Number of dimensions of y_train must be 1",
                         y_train_exception.exception.args[0])
        with self.assertRaises(ValueError) as train_data_exception:
            siamese_network.train_model(x_train, np.array([1, 2]), epochs=1)
            siamese_network.train_model(x_train, np.array([1, 2, 3, 4]), epochs=1)
        self.assertEqual("Number of samples in x_train and y_train must be the same",
                         train_data_exception.exception.args[0])
        with self.assertRaises(ValueError) as x_validation_data_exception:
            validation_data = (np.array([1, 2, 3, 4, 5, 6]), y_train)
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
            validation_data = (np.array([[1, 2], [3, 4]]), y_train)
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
        self.assertEqual("Shape of validation_data[0] must be (sample_size, 2, img_height, img_width, number_of_channels)",
                         x_validation_data_exception.exception.args[0])
        with self.assertRaises(ValueError) as y_validation_data_exception:
            validation_data = (x_train, np.array([[1, 2, 3], [4, 5, 6]]))
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
            validation_data = (x_train, np.array([[1], [2]]))
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
        self.assertEqual("Number of dimensions of validation_data[1] must be 1",
                         y_validation_data_exception.exception.args[0])
        with self.assertRaises(ValueError) as validation_data_exception:
            validation_data = (x_train, np.array([1, 2]))
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
            validation_data = (x_train, np.array([1, 2, 3, 4]))
            siamese_network.train_model(x_train, y_train, validation_data=validation_data, epochs=1)
        self.assertEqual("Number of samples in validation_data[0] and validation_data[1] must be the same",
                         validation_data_exception.exception.args[0])

    def test_train_model_arguments_values(self):
        x_train = np.array([
            [[[[1.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]],
            [[[[2.0]] * 10 for _ in range(10)], [[[2.0]] * 10 for _ in range(10)]],
            [[[[3.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]]
        ])
        y_train = np.array([1.0, 1.0, 2.0])
        siamese_network = SiameseConv2d((10, 10, 1))
        with self.assertRaises(ValueError) as epochs_exception:
            siamese_network.train_model(x_train, y_train, epochs=-10)
            siamese_network.train_model(x_train, y_train, epochs=0)
            siamese_network.train_model(x_train, y_train, epochs=-1)
        self.assertEqual("Argument 'epochs' must be greater than zero",
                         epochs_exception.exception.args[0])

    def test_train_model_output_weights(self):
        x_train = np.array([
            [[[[1.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]],
            [[[[2.0]] * 10 for _ in range(10)], [[[2.0]] * 10 for _ in range(10)]],
            [[[[3.0]] * 10 for _ in range(10)], [[[1.0]] * 10 for _ in range(10)]]
        ])
        y_train = np.array([1.0, 1.0, 2.0])
        siamese_network = SiameseConv2d((10, 10, 1))
        siamese_network.model.load_weights('test_weights_10x10.ckpt')
        tf.random.set_seed(1)
        history = siamese_network.train_model(x_train, y_train, verbose=0, epochs=10)
        expected_loss = [
            -0.050507064908742905, 0.1848236322402954,
            -0.11972253769636154, -0.2501557767391205,
            -0.20279167592525482, -0.07198188453912735,
            -0.2398887425661087, -0.09496841579675674,
            -0.21396155655384064, -0.1999618262052536
        ]
        self.assertEqual(history.history['loss'], expected_loss)

    def test_make_prediction_arguments_shapes(self):
        siamese_network = SiameseConv2d((50, 50, 1))
        image1 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image0.jpg')
        image1_array = preprocess_image(image1, 50, 50)
        image2 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image1.jpg')
        image2_array = preprocess_image(image2, 50, 50)
        with self.assertRaises(ValueError) as image1_error:
            siamese_network.make_prediction(preprocess_image(image1, 50, 40), image2_array)
            siamese_network.make_prediction(preprocess_image(image1, 20, 50), image2_array)
            siamese_network.make_prediction(preprocess_image(image1, 35, 75), image2_array)
        self.assertEqual("The shape of 'image1' and the shape "
                         "specified in the constructor of SiameseConv2d don't match", image1_error.exception.args[0])
        with self.assertRaises(ValueError) as image2_error:
            siamese_network.make_prediction(image1_array, preprocess_image(image1, 50, 40))
            siamese_network.make_prediction(image1_array, preprocess_image(image1, 20, 50))
            siamese_network.make_prediction(image1_array, preprocess_image(image1, 35, 75))
        self.assertEqual("The shape of 'image2' and the shape "
                         "specified in the constructor of SiameseConv2d don't match", image2_error.exception.args[0])

    def test_make_prediction_output_type(self):
        siamese_network = SiameseConv2d((50, 50, 1))
        siamese_network.model.load_weights('test_weights_50x50.ckpt')
        tf.random.set_seed(1)
        image1 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image0.jpg')
        image1 = preprocess_image(image1, 50, 50)
        image2 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image1.jpg')
        image2 = preprocess_image(image2, 50, 50)
        self.assertEqual(type(siamese_network.make_prediction(image1, image2)), np.float32)

    def test_make_prediction_output_value(self):
        siamese_network = SiameseConv2d((50, 50, 1))
        siamese_network.model.load_weights('test_weights_50x50.ckpt')
        tf.random.set_seed(1)
        image1 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image0.jpg')
        image1 = preprocess_image(image1, 50, 50)
        image2 = Image.open('C:/Work/PetProject/tests/test_images/0/test_image1.jpg')
        image2 = preprocess_image(image2, 50, 50)
        self.assertAlmostEqual(0.05911837, siamese_network.make_prediction(image1, image2), delta=0.001)


if __name__ == '__main__':
    main()