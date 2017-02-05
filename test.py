from detector import ObjectDetector
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--detector",required=True,help="path to trained detector to load...")
ap.add_argument("-i","--image",required=True,help="path to an image for object detection...")
ap.add_argument("-a","--annotate",default=None,help="text to annotate...")
args = vars(ap.parse_args())

detector = ObjectDetector(loadPath="detector.svm")

imagePath = args["image"]
image = cv2.imread(imagePath)
detector.detect(image,annotate=args["annotate"])
