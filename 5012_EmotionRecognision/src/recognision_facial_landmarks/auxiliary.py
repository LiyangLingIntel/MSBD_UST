#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "chen"

import cv2 as cv
import time
from PIL import Image
import numpy as np
import csv

class Mouth_Decector(object):

    def __init__(self):
        # load classifiers
        self.harr_face = cv.CascadeClassifier('../01_dataset_GENKI_4K/haarcascade_frontalface_default.xml')
        self.harr_mouth = cv.CascadeClassifier('../01_dataset_GENKI_4K/haarcascade_mouth.xml')


    # FA detected face as detected face
    def find_max_square(self, squares, tag='square', verbose=0):
        max_square_size = 0
        max_square = []
        if len(squares) > 0:
            # squares: [0]: x; [1]: y; [2]: width; [3]: height
            for (x, y, w, h) in squares:
                if  w * h > max_square_size:
                    max_square_size = w * h
                    max_square = [x, y, w, h]

            if verbose > 0:
                if len(max_square) == 0: # did not detect face
                    print('no %s found')# in %s' % (tag, img_path))
                else:
                    print('find %s @ %s' % (tag, str(max_square)))
        return max_square

    def find_lowest_square(self, squares, tag='square', verbose=0):
        max_y = 0
        lowest_square = []
        if len(squares) > 0:
            # squares: [0]: x; [1]: y; [2]: width; [3]: height
            for (x, y, w, h) in squares:
                if  y > max_y:
                    max_y = y
                    lowest_square = [x, y, w, h]

            if verbose > 0:
                if len(lowest_square) == 0: # did not detect face
                    print('no %s found in')# %s' % (tag, img_path))
                else:
                    print('find %s @ %s' % (tag, str(lowest_square)))

        return lowest_square

    def find_mouth(self, img, is_square_face=False, is_square_mouth=False):
        # runing the classifiers
    #     img = imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.harr_face.detectMultiScale(gray, 1.3, 5)
        max_face = self.find_max_square(faces, tag='face')
        # for (x, y, w, h) in faces:
        # img = cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        if len(max_face) > 0:
            x, y, w, h = max_face
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            if is_square_face:
                cv.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        #     imshow(img)
            mouths = self.harr_mouth.detectMultiScale(face_gray)
            mouth = self.find_lowest_square(mouths, tag='mouth')
        #     for (mx, my, mw, mh) in mouths:
        #         cv.rectangle(roi_color, (mx,my), (mx+mw, my+mh), (0,255,0), 1)
        #         print(mx, my, mw, mh)
            if len(mouth) > 0:
                mx, my, mw, mh = mouth
                if is_square_mouth:
                    cv.rectangle(face_color, (mx-1,my-1), (mx+mw+1, my+mh+1), (100,230,255), 2)
    #             imshow(img)
                return x+mx, y+my, mw, mh
        return []


    def get_partial(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]



    def normalize_img(self, img, is_grey=True, is_vectorize=False, width=28, height=10):
        size = width, height # (width, height)
        im = Image.fromarray(img)# Image.open(filename) 
        resized_im = im.resize(size, Image.ANTIALIAS) # resize image
        result = np.array(resized_im)
        if is_grey:
            im_grey = resized_im.convert('L') # convert the image to *greyscale*
            im_array = np.array(im_grey) # convert to np array
            result = im_array
        if is_vectorize:
            oned_array = result.reshape(size[0] * size[1])
            result = oned_array
        return result#np.array(resized_im)#oned_array
        
