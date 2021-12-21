import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("resources/ElonTain1.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon1 = face_recognition.load_image_file("resources/ElonTrain.jpg")
imgElon1 = cv2.cvtColor(imgElon1, cv2.COLOR_BGR2RGB)
imgElon2 = face_recognition.load_image_file("resources/ElonTest3.jfif")
imgElon2 = cv2.cvtColor(imgElon2, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (0,255,0),2)

faceloc1 = face_recognition.face_locations(imgElon1)[0]
encodeElon1 = face_recognition.face_encodings(imgElon1)[0]
cv2.rectangle(imgElon1, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (0,255,0),2)

faceloc2 = face_recognition.face_locations(imgElon2)[0]
encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
cv2.rectangle(imgElon2, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (0,255,0),2)

result = face_recognition.compare_faces([encodeElon, encodeElon1], encodeElon2)
faceDis = face_recognition.face_distance([encodeElon, encodeElon1], encodeElon2)
print(result, faceDis)

cv2.putText(imgElon2, f'{result} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)


# cv2.imshow("Elon Train", imgElon)
# cv2.imshow("Elon Test", imgElon1)
cv2.imshow("Elon Test", imgElon2)

cv2.waitKey(0)