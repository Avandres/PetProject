import os

from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

from SiameseNetworkModule import SiameseNetworkClass
from dataset_preprocess import DatasetPreprocessor

class ComputerVisionModule:

    def __init__(self,
                 siamese_network=None,
                 dataset_preprocessor=None,
                 owners_face_photo_directory=None):
        if siamese_network is None:
            siamese_network = SiameseNetworkClass((50, 50, 1))
            siamese_network.load_weights()
        self.siamese_network = siamese_network
        if dataset_preprocessor is None:
            dataset_preprocessor = DatasetPreprocessor()
        self.dataset_preprocessor = dataset_preprocessor
        self.owners_face_photo_directory = owners_face_photo_directory

    def is_owner(self, current_photo):
        current_photo = Image.fromarray(current_photo, 'RGB')
        current_photo = self.dataset_preprocessor.preprocess_image(current_photo, 50, 50)
        photos = os.listdir(self.owners_face_photo_directory)
        photo_distances_list = []
        for filename in photos:
            full_filename = self.owners_face_photo_directory + '//' + filename
            img = Image.open(full_filename)
            img_array = self.dataset_preprocessor.preprocess_image(img, 50, 50)
            photo_distances_list.append(self.siamese_network.make_prediction(current_photo, img_array))
        return photo_distances_list

if __name__ == '__main__':
    dataset_preprocessor = DatasetPreprocessor()
    network = SiameseNetworkClass((50, 50, 1))
    #x, y = dataset_preprocessor.get_data(50000, 50, 50, 'C://Work//PetProject//archive//Extracted Faces')
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #network.train_model(x_train, y_train, x_test, y_test)
    #network.save_weights('users_weights.ckpt')
    network.load_weights('users_weights.ckpt')


    cam = cv2.VideoCapture(0)
    result, photo = cam.read()
    cam.release()
    cv2.destroyAllWindows()
    module = ComputerVisionModule(siamese_network=network, owners_face_photo_directory='C://Work//PetProject//my_face')
    my_list = module.is_owner(photo)
    print(my_list)