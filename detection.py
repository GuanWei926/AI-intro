import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    with open(dataPath) as file:
        line_list = [line.rstrip() for line in file]
    line_idx = 0
    while line_idx < len(line_list):
        image_path = os.path.join(os.path.dirname(dataPath),line_list[line_idx]) 
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)
        num_faces = int(line_list[line_idx + 1])
        for i in range(num_faces):
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]

            left_top = (coord[0], coord[1])
            right_bottom = (coord[0]+coord[3], coord[1]+coord[2])

            img_crop = img_gray[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]].copy()
            img_crop = cv2.resize(img_crop, (19, 19))

            if clf.classify(img_crop):
                cv2.rectangle(img, (left_top[0], left_top[1]),(right_bottom[0], right_bottom[1]), (0, 255, 0),3)
            else:
                cv2.rectangle(img, (left_top[0], left_top[1]),(right_bottom[0], right_bottom[1]), (0, 0, 255),3)
        
        cv2.imshow(line_list[line_idx], img)  
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        line_idx += num_faces + 2
    # End your code (Part 4)
