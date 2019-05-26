from hyperlpr import *
import cv2

image = cv2.imread("demo.png")

print(HyperLPR_PlateRecogntion(image))
