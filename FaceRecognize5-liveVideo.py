from datetime import datetime
import face_recognition
import cv2
import os
import pickle
import time
import requests
print(cv2.__version__)

url='http://sriwijaya87.com/absen_jetson.php'
Encodings = []
Names = []

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(1)

while True:
    _, frame = cam.read()
    frameSmall = cv2.resize(frame, (0, 0), fx=.33, fy=.33)
    frameRGB = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB, model='cnn')
    allEncodings = face_recognition.face_encodings(frameRGB, facePositions)
    for (top, right, bottom, left), face_encoding in zip(facePositions, allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
            with open("D:/Kinan/Kuliah/Kelas 4/Skripsi/HMM/Program/AbsensiWajah/demoImages-master/absen.txt", "a+") as file_object:
                now = datetime.now()
                timee = now.strftime("%Y-%m-%d %H:%M:%S")
                file_object.write(name + "Absen pada : " + timee + "\n")
            print("Hallo : ", name)
            sendnama = {'nama_user_jetson': name}
            x = requests.post(url, data=sendnama)
            print(x.text)
        top = top * 3
        right = right * 3
        bottom = bottom * 3
        left = left * 3
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 6), font, .75, (0, 0, 255))
    cv2.imshow('Picture', frame)
    cv2.moveWindow('Picture', 0, 0)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
file_object.write("Session End\n")