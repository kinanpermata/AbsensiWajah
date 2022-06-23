from datetime import datetime
import face_recognition
import cv2
import os
import pickle
import time
print(cv2.__version__)
 
Encodings=[]
Names=[]
 
with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX

dispW = 320  # or 640 or 1280
dispH = 240  # or 480 or 960
flip = 2  # INverts image, otherwise upside down

camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=' + str(
    flip) + ' ! video/x-raw, width=' + str(dispW) + ', height=' + str(
    dispH) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cam= cv2.VideoCapture(camSet)
 
while True:
 
    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=.25,fy=.25)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
            with open("D:/Kinan/Kuliah/Kelas 4/Skripsi/HMM/Program/AbsensiWajah/demoImages-master/absen.txt", "a+") as file_object:
                now = datetime.now()
                timee = now.strftime("%Y-%m-%d %H:%M:%S")
                file_object.write(name + "Absen pada : " + timee + "\n")
            print("Halo : ", name)

        top=top*4
        right=right*4
        bottom=bottom*4
        left=left*4
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()
file_object.write("Session End\n")