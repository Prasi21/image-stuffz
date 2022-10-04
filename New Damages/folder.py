from distutils import extension
import random as rand
import os
import cv2 as cv
import glob

extension = ".png"
dirList = []

for file in os.listdir("temp"):
    if file.endswith(".png"):
        dirList.append(file)

# dirList[True for file in os.listdir("./temp/") if file.endswith(".txt") ]
sticker_name = rand.choice(dirList)
# sticker_name = rand.choice(os.listdir("./temp/")) # TODO skip gitignore?
sticker_path = os.path.join("temp",sticker_name)

sticker = cv.imread(sticker_path, cv.IMREAD_UNCHANGED)

print(sticker_path)
cv.imshow("rand", sticker)
cv.waitKey(0)