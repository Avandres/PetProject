import cv2

cam = cv2.VideoCapture(0)

result, image = cam.read()
print(image)
if result:
    cv2.imwrite('my_face/11.png', image)
cam.release()
cv2.destroyAllWindows()