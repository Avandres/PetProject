import cv2

cam = cv2.VideoCapture(0)

result, image = cam.read()
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
)

print("Found {0} Faces!".format(len(faces)))
for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)
    print(x, y, w, h)
    x1, x2 = x, x + w
    y1, y2 = y, y + h
    print(x1, y1, x2, y2)
    print(image.shape)
    image = image[y1:y2, x1:x2, :]
    print(image.shape)
    #print(image)
    cv2.imwrite('my_face/10.png', image)

cam.release()