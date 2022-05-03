from pickletools import optimize
import numpy as np
import matplotlib as plt
import cv2 as cv
from PIL import Image

img = Image.open("./image stuffz/clean_image.jpg")

img.thumbnail(img.size, Image.ANTIALIAS)
img.save("./image stuffz/AA_sign.jpg", quality = 100, optimize = True)