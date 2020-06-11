import cv2


class CEye:
    def __init__(self, img, direction, x):
        self.cgray = cv2.resize(img, (50, 42), interpolation=cv2.INTER_AREA)
        # self.cgray = cv2.cvtColor(self.ccolor, cv2.COLOR_BGR2GRAY)
        self.cclass = direction
        self.cwhich = []
        self.x = x

    def add_lr(self, lr):
        self.cwhich = lr