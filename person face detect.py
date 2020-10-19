import numpy as np
import cv2
import face_recognition
imgPrim = face_recognition.load_image_file('img\haroon.jpg')
imgPrim = cv2.cvtColor(imgPrim, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file('img/1cffb311-d56e-4acc-bc5f-15712f11a796.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgPrim)[0]
faceEncod = face_recognition.face_encodings(imgPrim)[0]
cv2.rectangle(imgPrim, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), color=(255, 0, 255), thickness=2)

faceLocTest = face_recognition.face_locations(img_test)[0]
faceEncodTest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), color=(255, 0, 255), thickness=2)
result = face_recognition.compare_faces([faceEncod], faceEncodTest)
distance = face_recognition.face_distance([faceEncod], faceEncodTest)

cv2.putText(img_test, f'{result} {round(distance[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1 ,(255, 0, 255), 4)
print(result, distance)


cv2.imshow('haroon', imgPrim)
cv2.imshow('test', img_test)
cv2.waitKey(0) & 0xff == 'q'
cv2.destroyAllWindows()
