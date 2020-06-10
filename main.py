# import dataloader

import os
import cv2
import os
import glob


# ladowanie zdjec jednej klasy


class CImage:
    def __init__(self, path):
        self.ccolor = cv2.imread(path)
        self.cgray = cv2.cvtColor(self.ccolor, cv2.COLOR_BGR2GRAY)
        self.cfacecord = []
        self.croi_gray = []
        self.croi_color = []
        self.ceyecord = []  # np.ones((1, 4), dtype = int)


class CEye:
    def __init__(self, img, direction, x):
        self.ccolor = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (50, 42), interpolation=cv2.INTER_AREA)
        self.cclass = direction
        self.cwhich = []
        self.x = x

    def add_lr(self, lr):
        self.cwhich = lr


def loadfiles(img_dir):
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    # i = 0
    for f1 in files:
        img = CImage(f1)
        data.append(img)
        # i += 1
        # if i > 49:
        #     break

    return data


def extract_eyes(images):
    eyes = []

    for img in images:
        img.cfacecord = face_cascade.detectMultiScale(img.cgray, 1.3, 5)
        for (x, y, w, h) in img.cfacecord:
            cv2.rectangle(img.ccolor, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img.roi_gray = img.cgray[y:y + h, x:x + w]
            img.roi_color = img.ccolor[y:y + h, x:x + w]
            img.ceyecord = eye_cascade.detectMultiScale(img.roi_gray)
            eyecount = 0
            for (ex, ey, ew, eh) in img.ceyecord:
                cv2.rectangle(img.roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyecount += 1
                if eyecount == 1:
                    eye1 = CEye(img.roi_color[ey:ey + eh, ex:ex + ew], 'center', ex)
                if eyecount == 2:
                    eye2 = CEye(img.roi_color[ey:ey + eh, ex:ex + ew], 'center', ex)
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


# Ladowanie klasyfikatorow

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Pusta lista na wykryte oczy
eyes = []

# Wywolanie funkcji dataloadera
images = loadfiles("Eye_chimeraToPublish/00.Centre/")

# wywolanie funkcji ekstrakcji oczu
eyes = extract_eyes(images)

# wyświetlenie obrazu

for img in eyes:
    cv2.putText(img.ccolor, img.cwhich, (5, 10), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
    cv2.imshow('img', img.ccolor)

    k = cv2.waitKey(0)
    if k == 27:
        break
    # zamknięcie okna
cv2.destroyAllWindows()
# oczekiwanie na znak z klawiatury ESC wyłącza okno
