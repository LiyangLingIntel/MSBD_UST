#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "chen"

import cv2 as cv
import time
from PIL import Image
import numpy as np
import csv
import dlib
import math
from keras.models import load_model
# import mxnet as mx
# from mtcnn_detector import MtcnnDetector

class Mouth_Decector(object):

    def __init__(self):
        # load classifiers
        self.harr_face = cv.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
        # self.harr_mouth = cv.CascadeClassifier('./model/haarcascade_mouth.xml')
        # self.detector = dlib.get_frontal_face_detector()
        self.face_rec_model = load_model('./model/model_face_rec.h5')
        # self.detector_mx = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark = False)

    def normalize_img(self, img, is_grey=True, is_vectorize=False, width=28, height=10, is_standardrize=False):
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
        if is_standardrize:
            result = result / 255
        return result#np.array(resized_im)#oned_array

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

    def get_inver_norm_pts(self, img_lb, width=96, height=90, is_unstd=True):
        img_lb_xs = img_lb[np.arange(0, 9, 2)]
        img_lb_ys = img_lb[np.arange(1, 11, 2)]

        pts = []
        for x,y in zip(img_lb_xs, img_lb_ys):
            if is_unstd:
                x *= width
                y *= height
            x = int(round(x))
            y = int(round(y))
            pts.append([x, y])
        pts = np.array(pts)
        return pts

    def get_mouth_box(self, pts):
        if isinstance(pts, np.ndarray) and pts.shape == (5,2):
            p = pts[:,0].tolist()
            p.extend(pts[:,1].tolist())
            pts = p
        if len(pts) < 10:
            return []
        box=[]
        height = abs(pts[2+5]-pts[4+5])
        width = abs(pts[4]-pts[3])
        x = pts[3]*.95
        y = pts[3+5] - height/2*0.8
        w = width*1.1
        h = height*1.3
        box.append(int(math.floor(x)))
        box.append(int(math.floor(y)))
        box.append(int(math.ceil(w)))
        box.append(int(math.ceil(h)))
        return box
 

    def find_mouth(self, img, width=96, height=90):
        img_org = img
        h = img.shape[0]
        w = img.shape[1]
        img = self.normalize_img(img, is_grey=True, is_vectorize=False, width=width, height=height, is_standardrize=True)
        img_lb = self.face_rec_model.predict(img.reshape(-1, height, width, 1)).reshape(-1)
        pts = self.get_inver_norm_pts(img_lb, width=w, height=h, is_unstd=True)
        fx, fy, fw, fh = self.get_mouth_box(pts)
        return (fx, fy, fw, fh), pts


    def find_mouths(self, img, is_square_face=False, is_square_mouth=False):
        # faces = self.harr_face.detectMultiScale(gray, 1.3, 5)
        min_face = 4000
        # for (x, y, w, h) in faces:
        # img = cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        found_mouths=[]

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for face in self.harr_face.detectMultiScale(image = gray, scaleFactor = 1.3, minNeighbors = 3):
        # results = self.detector_mx.detect_face(img)
        # if results is None:
        #     return found_mouths
        # for face_det in results[0]:
        #     face = [face_det[0], face_det[1], face_det[2] - face_det[0], face_det[4] - face_det[2]]
            # face = [face_det.left(), face_det.top(), face_det.right() - face_det.left(), face_det.bottom() - face_det.top()]
            if face[2]*face[3] > min_face:
                x, y, w, h = face
                face_gray = gray[y:y+h, x:x+w]
                face_color = img[y:y+h, x:x+w]
                if is_square_face:
                    cv.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)
                    # (fx, fy, fh, fw) , pts = self.find_face(face_color)
                    # img = cv.rectangle(face_color, (fx,fy), (fx+fw, fy+fh), (255,255,0), 2)
                    mouth , pts = self.find_mouth(face_color)
                    for i, pt in enumerate(pts):
                        cv.circle(face_color, (pt[0], pt[1]), 2, (0,0,255), 2)
                # mouths = self.harr_mouth.detectMultiScale(face_gray)
                # mouth = self.find_lowest_square(mouths, tag='mouth')
            #     for (mx, my, mw, mh) in mouths:
            #         cv.rectangle(roi_color, (mx,my), (mx+mw, my+mh), (0,255,0), 1)
            #         print(mx, my, mw, mh)
 
                if len(mouth) > 0:
                    mx, my, mw, mh = mouth
                    if is_square_mouth:
                        cv.rectangle(face_color, (mx-1,my-1), (mx+mw+1, my+mh+1), (100,230,255), 2)
        #             imshow(img)
                    found_mouths.append([[x+mx, y+my, mw, mh],w])
                        # yield x+mx, y+my, mw, mh
        return found_mouths
        # return []


    def get_partial(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]
