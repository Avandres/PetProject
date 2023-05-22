import os

from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

from siamese_networks import SiameseConv2d
from dataset_preprocess import DatasetPreprocessor


class ComputerVisionModule:

    def __init__(self,
                 siamese_network=None,
                 dataset_preprocessor=None,
                 owners_face_photo_directory=None):
        if siamese_network is None:
            siamese_network = SiameseConv2d((50, 50, 1))
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
    network = SiameseConv2d((50, 50, 1))
    #x, y = dataset_preprocessor.get_data(150000, 50, 50, 'C://Work//PetProject//archive//Extracted Faces')
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    network.load_weights('users_weights.ckpt')
    #network.train_model(x_train, y_train, x_test, y_test, epochs=2000)
    #network.save_weights('users_weights2.ckpt')

    cam = cv2.VideoCapture(0)
    result, photo = cam.read()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        photo,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    print("Found {0} Faces!".format(len(faces)))
    x, y, w, h = faces[0]
    x1, x2 = x, x + w
    y1, y2 = y, y + h
    photo = photo[y:y + h, x:x + w, :]
    cam.release()

    module = ComputerVisionModule(siamese_network=network, owners_face_photo_directory='C://Work//PetProject//my_face')
    my_list = module.is_owner(photo)
    print(my_list)

    # for i in os.listdir('C://Work//PetProject//archive//Extracted Faces'):
    #     photo = cv2.imread('C://Work//PetProject//archive//Extracted Faces//' + i + '//0.jpg')
    #     my_list = module.is_owner(photo)
    #     for j in my_list:
    #         if j < 0.4:
    #             print(i + '//' + '0.jpg     ', True)