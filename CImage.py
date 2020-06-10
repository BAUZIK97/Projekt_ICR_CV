import cv2

class CImage:
    def __init__(self, path, imgdir):
        self.ccolor = cv2.imread(path)
        self.cgray = cv2.cvtColor(self.ccolor, cv2.COLOR_BGR2GRAY)
        self.cfacecord = []
        self.croi_gray = []
        self.croi_color = []
        self.cclass = imgdir[3:len(imgdir)-1]
        self.ceyecord = []  # np.ones((1, 4), dtype = int)

