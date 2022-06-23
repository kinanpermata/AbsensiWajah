import pickle
import face_recognition
import cv2
import os
print(cv2.__version__)

Encodings=[]
Names=[]

image_dir='D:/Kinan/Kuliah/Kelas 4/Skripsi/HMM/Program/AbsensiWajah/demoImages-master/known'
for root, dirs, files in os.walk(image_dir):
    print(files)
    for file in files:
        path=os.path.join(root,file)
        print(path)
        name=os.path.splitext(file)[0]
        print(name)
        person=face_recognition.load_image_file(path)
        encodings=face_recognition.face_encodings(person)[0]
        Encodings.append(encodings)
        Names.append(name)
print(Names)

with open('train.pkl','wb') as f:
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)