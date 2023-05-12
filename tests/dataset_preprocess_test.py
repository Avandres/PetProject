from unittest import TestCase, main
import os

from PIL import Image
import numpy as np

from dataset_preprocess import DatasetPreprocessor


class DatasetPreprocessorTest(TestCase):

    def test_preprocess_image_is_normalized(self):
        img = Image.open('test_images/0/test_image0.jpg')
        test_dataset_preprocessor = DatasetPreprocessor()
        preprocessed_image = test_dataset_preprocessor.preprocess_image(img, 50, 50)
        self.assertTrue((preprocessed_image <= 1.0).all())
        self.assertTrue((preprocessed_image >= 0.0).all())

    def test_preprocess_image_input_types(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        with self.assertRaises(TypeError) as img_type_error:
            test_dataset_preprocessor.preprocess_image(np.zeros((50, 50)), 50, 50)
            test_dataset_preprocessor.preprocess_image(14, 50, 50)
            test_dataset_preprocessor.preprocess_image('img', 50, 50)
            test_dataset_preprocessor.preprocess_image(True, 50, 50)
        self.assertEqual("Argument 'img' must be PIL.Image.", img_type_error.exception.args[0])
        img = Image.open('test_images/0/test_image0.jpg')
        with self.assertRaises(TypeError) as height_type_error:
            test_dataset_preprocessor.preprocess_image(img, 'str', 50)
            test_dataset_preprocessor.preprocess_image(img, 4.25, 50)
            test_dataset_preprocessor.preprocess_image(img, True, 50)
            test_dataset_preprocessor.preprocess_image(img, [1, 2, 3], 50)
        self.assertEqual("Argument 'height' must be int.", height_type_error.exception.args[0])
        with self.assertRaises(TypeError) as width_type_error:
            test_dataset_preprocessor.preprocess_image(img, 50, 'str')
            test_dataset_preprocessor.preprocess_image(img, 50, 4.25)
            test_dataset_preprocessor.preprocess_image(img, 50, True)
            test_dataset_preprocessor.preprocess_image(img, 50, [1, 2, 3])
        self.assertEqual("Argument 'width' must be int.", width_type_error.exception.args[0])

    def test_preprocess_image_shape(self):
        img = Image.open('test_images/0/test_image0.jpg')
        test_dataset_preprocessor = DatasetPreprocessor()
        preprocessed_image = test_dataset_preprocessor.preprocess_image(img, 50, 50)
        self.assertEqual(preprocessed_image.shape, (50, 50))
        preprocessed_image = test_dataset_preprocessor.preprocess_image(img, 24, 103)
        self.assertEqual(preprocessed_image.shape, (24, 103))
        preprocessed_image = test_dataset_preprocessor.preprocess_image(img, 5, 10)
        self.assertEqual(preprocessed_image.shape, (5, 10))
        with self.assertRaises(ValueError) as img_height_error:
            test_dataset_preprocessor.preprocess_image(img, 0, 50)
            test_dataset_preprocessor.preprocess_image(img, -1, 50)
        self.assertEqual("Argument 'img_height' must be greater than zero.", img_height_error.exception.args[0])
        with self.assertRaises(ValueError) as img_width_error:
            test_dataset_preprocessor.preprocess_image(img, 50, 0)
            test_dataset_preprocessor.preprocess_image(img, 50, -1)
        self.assertEqual("Argument 'img_width' must be greater than zero.", img_width_error.exception.args[0])

    def test_get_data_sample_size(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        x_data, y_data = test_dataset_preprocessor.get_data(4, 50, 50, path='test_images')
        self.assertEqual(x_data.shape, (4, 2, 50, 50, 1))
        self.assertEqual(y_data.shape, (4, 1))
        x_data, y_data = test_dataset_preprocessor.get_data(1, 50, 50, path='test_images')
        self.assertEqual(x_data.shape, (1, 2, 50, 50, 1))
        self.assertEqual(y_data.shape, (1, 1))
        with self.assertRaises(ValueError) as sample_size_error:
            test_dataset_preprocessor.get_data(0, 50, 50, path='test_images')
            test_dataset_preprocessor.get_data(-1, 50, 50, path='test_images')
        self.assertEqual("Argument 'total_sample_size' must be greater than zero.", sample_size_error.exception.args[0])

    def test_get_data_arrays_shape(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        x_data, y_data = test_dataset_preprocessor.get_data(4, 50, 50, path='test_images')
        self.assertEqual(x_data.shape, (4, 2, 50, 50, 1))
        x_data, y_data = test_dataset_preprocessor.get_data(4, 39, 42, path='test_images')
        self.assertEqual(x_data.shape, (4, 2, 39, 42, 1))
        with self.assertRaises(ValueError) as img_height_error:
            test_dataset_preprocessor.get_data(1, 0, 50, path='test_images')
            test_dataset_preprocessor.get_data(1, -1, 50, path='test_images')
        self.assertEqual("Argument 'img_height' must be greater than zero.", img_height_error.exception.args[0])
        with self.assertRaises(ValueError) as img_width_error:
            test_dataset_preprocessor.get_data(1, 50, 0, path='test_images')
            test_dataset_preprocessor.get_data(1, 50, -1, path='test_images')
        self.assertEqual("Argument 'img_width' must be greater than zero.", img_width_error.exception.args[0])

    def test_get_data_path(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        with self.assertRaises(FileNotFoundError) as path_error:
            test_dataset_preprocessor.get_data(4, 50, 50, path=os.getcwd())
        self.assertEqual("There is no *.png or *.jpg files in " + os.getcwd() + '.', path_error.exception.args[0])

    def test_get_data_input_types(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        with self.assertRaises(TypeError) as sample_size_type_error:
            test_dataset_preprocessor.get_data(np.zeros((50, 50)), 50, 50)
            test_dataset_preprocessor.get_data('img', 50, 50)
            test_dataset_preprocessor.get_data(True, 50, 50)
        self.assertEqual("Argument 'total_sample_size' must be int.", sample_size_type_error.exception.args[0])
        with self.assertRaises(TypeError) as height_type_error:
            test_dataset_preprocessor.get_data(4, 'str', 50)
            test_dataset_preprocessor.get_data(4, 4.25, 50)
            test_dataset_preprocessor.get_data(4, True, 50)
        self.assertEqual("Argument 'img_height' must be int.", height_type_error.exception.args[0])
        with self.assertRaises(TypeError) as width_type_error:
            test_dataset_preprocessor.get_data(4, 50, 'str')
            test_dataset_preprocessor.get_data(4, 50, 4.25)
            test_dataset_preprocessor.get_data(4, 50, True)
        self.assertEqual("Argument 'img_width' must be int.", width_type_error.exception.args[0])
        with self.assertRaises(TypeError) as path_type_error:
            test_dataset_preprocessor.get_data(4, 50, 50, path=1)
            test_dataset_preprocessor.get_data(4, 50, 50, path=1.1)
            test_dataset_preprocessor.get_data(4, 50, 50, path=True)
        self.assertEqual("Argument 'path' must be int.", path_type_error.exception.args[0])

    def test_get_data_output_types(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        x_data, y_data = test_dataset_preprocessor.get_data(4, 50, 50, path='test_images')
        self.assertEqual(x_data.dtype, float)
        self.assertEqual(y_data.dtype, float)


if __name__ == '__main__':
    main()