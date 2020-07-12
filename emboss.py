
import cv2
import os
import numpy as np

# Pre process training data
for subdir, dirs, files in os.walk("./img/x"):
    for file in files:
        print(file)
        filepath = subdir + os.sep + file
        img = cv2.imread(filepath)
        # Apply actual emboss effect
        img = cv2.filter2D(img, -1, np.array([
            [0, -1, -1],
            [1, 0, -1],
            [1, 1, 0]]))
        cv2.imwrite("./img/y/"+file, img)
