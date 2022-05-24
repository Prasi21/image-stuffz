from pickletools import optimize
import numpy as np
import matplotlib as plt
import cv2 as cv
from PIL import Image

#attempt at opening image at a larger size and the tring to rescale it and use Anti aliasing features
#this approach is a dead end probably

img = Image.open("./image stuffz/clean_image.jpg")

img.thumbnail(img.size, Image.ANTIALIAS)
img.save("./image stuffz/AA_sign.jpg", quality = 100, optimize = True)