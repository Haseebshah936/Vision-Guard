import cv2
import numpy as np
import face_recognition
import os
import firebase_admin
from datetime import datetime
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./ServiceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


today = datetime.today()
d2 = today.strftime("%B %d %Y")
print("d2 =", d2)
attendancePath = "Attendance"+"/"+d2+".csv"
print(attendancePath)
if(os.path.exists(attendancePath)):
    print('true')
else:
    file = open(attendancePath,'w+')
    file.write("Email,Time")



def markattendence(email):
    with open(attendancePath, 'r+') as f:
        myDataList = f.readlines()
        emailList = []
        for line in myDataList:
            entry = line.split(',')
            emailList.append(entry[0])
        if email not in emailList:
            now = datetime.now()
            ref = db.collection(u'attendance')
            ref.add({"entryTime": firestore.SERVER_TIMESTAMP, "present": True, "email": email})
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{email},{dtString}')

path = "TrainingData"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for imName in myList:
    curImg = cv2.imread(f'{path}/{imName}')
    images.append(curImg)
    classNames.append(os.path.splitext(imName)[0])
print(classNames)

def findEncoding(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

encodings = findEncoding(images)
# print(encodings)

cp = cv2.VideoCapture(0)

cp.set(10, 100)

cap = cv2.VideoCapture(0)



while True:
    # success, img = cp.read()
    # imgS = cv2.resize(img,(0,0),None,1,1)
    # # imgS = cv2.resize(img,(0,0),None,0.5,0.5)
    # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    success2, img2 = cap.read()
    imgS2 = cv2.resize(img2, (0, 0), None, 1, 1)
    # imgS2 = cv2.resize(img2,(0,0),None,0.5,0.5)
    imgS2 = cv2.cvtColor(imgS2, cv2.COLOR_BGR2RGB)

    # faceloc = face_recognition.face_locations(imgS)
    # encoding = face_recognition.face_encodings(imgS, faceloc)
    faceloc2 = face_recognition.face_locations(imgS2)
    encoding2 = face_recognition.face_encodings(imgS2, faceloc2)



    # for encode, facelo in zip(encoding, faceloc):
    #     matches = face_recognition.compare_faces(encodings, encode)
    #     matchdistances = face_recognition.face_distance(encodings, encode)
    #     matchIndex = np.argmin(matchdistances)
    #     y1, x2, y2, x1 = facelo
    #     # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    #     # y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #     if matches[matchIndex]:
    #         name = classNames[matchIndex].upper()
    #         cv2.putText(img,name, (x1+6, y2+25), cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 0, 255),1)

    for encode, facelo in zip(encoding2, faceloc2):
        matches = face_recognition.compare_faces(encodings, encode)
        matchdistances = face_recognition.face_distance(encodings, encode)
        matchIndex = np.argmin(matchdistances)
        y1, x2, y2, x1 = facelo
        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        # y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if matches[matchIndex]:
            email = classNames[matchIndex].lower()
            entry = email.split('@')
            cv2.putText(img2,entry[0], (x1, y2+15), cv2.FONT_HERSHEY_COMPLEX,0.3,(0, 255, 255),1)
            markattendence(email)

    # cv2.imshow("Camera",img)
    cv2.imshow("Camera2", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

