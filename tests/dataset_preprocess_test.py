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
        self.assertEqual("Argument 'height' must be greater than zero.", img_height_error.exception.args[0])
        with self.assertRaises(ValueError) as img_width_error:
            test_dataset_preprocessor.preprocess_image(img, 50, 0)
            test_dataset_preprocessor.preprocess_image(img, 50, -1)
        self.assertEqual("Argument 'width' must be greater than zero.", img_width_error.exception.args[0])

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
        with self.assertRaises(NotADirectoryError) as path_error:
            test_dataset_preprocessor.get_data(4, 50, 50, path="test_hollow_directory")
        self.assertEqual("There are no directories on the specified path: test_hollow_directory", path_error.exception.args[0])
        with self.assertRaises(FileNotFoundError) as path_error:
            test_dataset_preprocessor.get_data(4, 50, 50, path="test_directory_with_hollow_directories")
        self.assertEqual(
            "There are no *.png or *.jpg files on the specified path: " +
            os.path.join("test_directory_with_hollow_directories", "hollow_0"),
            path_error.exception.args[0]
        )

    def test_get_data_output_types(self):
        test_dataset_preprocessor = DatasetPreprocessor()
        x_data, y_data = test_dataset_preprocessor.get_data(4, 50, 50, path='test_images')
        self.assertEqual(x_data.dtype, float)
        self.assertEqual(y_data.dtype, float)


if __name__ == '__main__':
    main()