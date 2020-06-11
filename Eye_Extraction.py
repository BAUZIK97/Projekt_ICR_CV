# import dataloader

import os
import cv2
import os
import glob
from CImage import CImage
from CEye import CEye
from pathlib import Path


# ladowanie zdjec jednej klasy
def loadfiles(img_dir):
    data_path = os.path.join('Eye_chimeraToPublish/'+img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    # i = 0
    for f1 in files:
        img = CImage(f1, img_dir)
        data.append(img)
        # i += 1
        # if i > 49:
        #     break

    return data


def haar_extract_eyes(images):
    eyes = []
    # Petla iterujaca po zdjeciach wejsciowych
    for img in images:
        # wywolanie detekcji kaskadowej twarzy
        img.cfacecord = face_cascade.detectMultiScale(img.cgray, 1.3, 5)
        # Wydzielanie obrebu twarzy z obrazu wejsciowego
        for (x, y, w, h) in img.cfacecord:

            img.roi_gray = img.cgray[y:y + h, x:x + w]
            img.ceyecord = eye_cascade.detectMultiScale(img.roi_gray, scaleFactor=1.1, minNeighbors=3)
            eyecount = 0
            for (ex, ey, ew, eh) in img.ceyecord:
                # cv2.rectangle(img.roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyecount += 1
                if eyecount == 1:
                    eye1 = CEye(img.roi_gray[ey:ey + eh, ex:ex + ew], img.cclass, ex)
                if eyecount == 2:
                    eye2 = CEye(img.roi_gray[ey:ey + eh, ex:ex + ew], img.cclass, ex)
                    if eye1.x > eye2.x:
                        eye1.add_lr('left')
                        eye2.add_lr('right')
                    else:
                        eye1.add_lr('right')
                        eye2.add_lr('left')
                        eyes.append(eye1)
                        eyes.append(eye2)
                    break
    return eyes


def savedata(eyes):
    count = 0
    for img in eyes:
        # Tworzenie sciezki docelowej
        if count < 0.6*len(eyes):
            train_val_test = 'train/'
        elif count < 0.8*len(eyes):
            train_val_test = 'val/'
        else :
            train_val_test = 'test/'

        path = 'Output_haar/' + str(img.cwhich) + '/' + str(train_val_test) + img.cclass + '/'
        if not os.path.isdir(path):
            path1 = Path(path)
            path1.mkdir(parents=True)
        # Zapisanie uzyskanego obrazu oka w podanym folderze
        cv2.imwrite(path + '/' + str(count) + '.jpg', img.cgray)
        count += 1


def loadclass(directory):
    images = loadfiles(directory)
    eyes = haar_extract_eyes(images)
    savedata(eyes)


# Ladowanie klasyfikatorow

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print('Classifier succestully loaded')

loadclass('00.Centre/')
print('Centre succesfully extracted')
loadclass('01.UpRight/')
print('UpRight succesfully extracted')
loadclass('02.Upleft/')
print('Upleft succesfully extracted')
loadclass('03.Right/')
print('Right succesfully extracted')
loadclass('04.Left/')
print('Left succesfully extracted')
loadclass('05.DownRight/')
print('DownRight succesfully extracted')
loadclass('06.Downleft/')
print('Downleft succesfully extracted')
