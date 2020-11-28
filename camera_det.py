import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from pathlib import Path
import glob

# ladowanie klasyfikatorow
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# przechwytywanie obrazu oraz wybor modelu
cap = cv2.VideoCapture(0)
classes = ['Centre', 'Left', 'Right']
choosemodel = int(input('Choose model: \n 0 - chimera trained \n 1 - camera trained'))
if choosemodel == 0:
    model_left = load_model('3classmodel_left.h5')
    model_right = load_model('3classmodel_right.h5')
else:
    model_left = load_model('3classmodel_left_camera.h5')
    model_right = load_model('3classmodel_right_camera.h5')
#parametry tekstu
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 20)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def loadfiles1(img_dir):
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1, cv2.IMREAD_COLOR)
        data.append(img)
    return data


def writetext(img, text, bottomLeftCornerOfText):
    # funkcja wyswietlajaca tekst
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


# deklaracje zmiennych
train_val_set = 'train'
old_eye = np.zeros((42, 50, 3), dtype=int)
new_eye = np.zeros((42, 50, 3), dtype=int)
right_eye = np.zeros((42, 50, 3), dtype=int)
left_eye = np.zeros((42, 50, 3), dtype=int)
eye = [old_eye, new_eye]
test = []
flag1 = 0
loopcounter = 0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        eyecount = 0
        ex1 = 0

        for (ex , ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eyecount += 1
            if eyecount == 1:
                eye1 = roi_gray[ey:ey + eh, ex:ex + ew]
                eye1 = cv2.cvtColor(eye1, cv2.COLOR_GRAY2BGR)
                eye1 = cv2.resize(eye1, (50, 42), interpolation=cv2.INTER_AREA)
                ex1 = ex
            if eyecount == 2:
                eye2 = roi_gray[ey:ey + eh, ex:ex + ew]
                eye2 = cv2.cvtColor(eye2, cv2.COLOR_GRAY2BGR)
                eye2 = cv2.resize(eye2, (50, 42), interpolation=cv2.INTER_AREA)
                ex2 = ex
                if ex1 > ex2:
                    new_eye = eye2
                    left_eye = eye2
                    right_eye = eye1
                else:
                    new_eye = eye1
                    left_eye = eye1
                    right_eye = eye2
                break
    test = np.array(left_eye, ndmin=4)
    test = test / 255
    predictions = np.argmax(model_left.predict(test), axis=-1)
    writetext(img, classes[predictions[0]], (10, 20))
    print('left' + str(classes[predictions[0]]))
    test = np.array(right_eye, ndmin=4)
    test = test / 255
    predictions = np.argmax(model_left.predict(test), axis=-1)
    writetext(img, classes[predictions[0]], (300, 20))
    print('right' + str(classes[predictions[0]]))
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
# out.release()
cv2.destroyAllWindows()
