import cv2


class CEye:
    def __init__(self, img, direction, x):
        self.ccolor = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (50, 42), interpolation=cv2.INTER_AREA)
        self.cclass = direction
        self.cwhich = []
        self.x = x

    def add_lr(self, lr):
        self.cwhich = lr