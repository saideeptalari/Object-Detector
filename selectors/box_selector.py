import numpy as np
import cv2

class BoxSelector(object):
    def __init__(self, image, window_name,color=(0,0,255)):
        #store image and an original copy
        self.image = image
        self.orig = image.copy()

        #capture start and end point co-ordinates
        self.start = None
        self.end = None

        #flag to indicate tracking
        self.track = False
        self.color = color
        self.window_name = window_name

        #hook callback to the named window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,self.mouseCallBack)

    def mouseCallBack(self, event, x, y, flags, params):
        #start tracking if left-button-clicked down
        if event==cv2.EVENT_LBUTTONDOWN:
            self.start = (x,y)
            self.track = True

        #capture/end tracking while mouse-move or left-button-click released
        elif self.track and (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONUP):
            self.end = (x,y)
            if not self.start==self.end:
                self.image = self.orig.copy()
                #draw rectangle on the image
                cv2.rectangle(self.image, self.start, self.end, self.color, 2)
                if event==cv2.EVENT_LBUTTONUP:
                    self.track=False

            #in case of clicked accidently, reset tracking
            else:
                self.image = self.orig.copy()
                self.start = None
                self.track = False
            cv2.imshow(self.window_name,self.image)

    @property
    def roiPts(self):
        if self.start and self.end:
            pts = np.array([self.start,self.end])
            s = np.sum(pts,axis=1)
            (x,y) = pts[np.argmin(s)]
            (xb,yb) = pts[np.argmax(s)]
            return [(x,y),(xb,yb)]
        else:
            return []


