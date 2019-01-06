# %matplotlib inline
import os
import math
import numpy as np
import pandas as pd
from os.path import join as pjoin

import matplotlib.pyplot as plt
import cv2 as cv
from skimage.io import imshow, imread, imsave
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

work_dir = '.'
file_path = './face/'
label_data = './labels.csv'

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import normalize

pd_face = pd.read_csv('./face_index.csv')#, index_col=0)

import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2


def find_pts(img, verbose=0, is_show=False, is_rect=False, is_normalize=False):
    #     img = imread(pd_face.iloc[9]['path'])
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    results = detector.detect_face(img)
    if results is not None:
        total_boxes = results[0]
        pts = results[1]
        print(pts)
    draw = img.copy()
    if verbose > 0:
        print(f"image shape: {str(img.shape)}")
        #         print("Number of faces detected: {}".format(len(dets)))

        for i in range(len(total_boxes)):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
                i, total_boxes[i][0], total_boxes[i][1], total_boxes[i][2], total_boxes[i][3]))
            print(f"aspect_ratio:{(total_boxes[i][3] - total_boxes[i][1]) / (total_boxes[i][2] -  total_boxes[i][0])}")
    if is_rect:
        img = cv.rectangle(draw, (int(total_boxes[i][0]), int(total_boxes[i][1])),
                           (int(total_boxes[i][2]), int(total_boxes[i][3])), (255, 0, 0), 1)
    #     shape = predictor(img, face)
    # use 15 points
    for i in range(len(pts)):
        if is_rect:
            for j in range(5):
                cv2.circle(draw, (pts[i][j], pts[i][j + 5]), 1, (0, 0, 255), 2)
        #     for i in range(5):#[17, 21, 22, 26, 30, 36, 39, 41, 42, 46, 47, 49, 52, 55, 58]:#range(68):
        #         if is_rect:
        #             img = cv.circle(img.copy(), (shape.part(i).x, shape.part(i).y), 1, (0,0,255), 1)
        # #                 cv.putText(img,str(i), (shape.part(i).x,shape.part(i).y), cv.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0), 1)
        #         x = shape.part(i).x
        #         y = shape.part(i).y
        # #             print(f"{x}, {y}")
        #         if is_normalize:
        #             x = shape.part(i).x / img.shape[0]
        #             y = shape.part(i).y / img.shape[1]
        #         pts.append(x)
        #         pts.append(y)

        break

    if is_show:
        cv2.imshow('image',draw)
        cv2.waitKey(0)

    return pts


def get_face_mouth_position(img_gen, is_compress=False, width=96, height=90,
                            verbos=1, pt_length=5, is_debug=False, is_normalize=False):
    columns = [x for x in range(pt_length)]
    columns.insert(0, 'index')
    columns.insert(1, 'path')
    pts = pd.DataFrame(columns=columns)

    for index, img_path in img_gen:
        if is_debug and index > 20:
            break
        try:
            if is_compress:
                img = normalize_img(img_path, is_grey=True, is_vectorize=False,
                                    width=width, height=height)
            else:
                img = imread(img_path)
                img = img.copy()
            #             face, mouth = find_face_mouth(img=img)

            fpt = find_pts(img=img, is_normalize=is_normalize)

            if len(fpt) > 0:
                #                 print(len(fpt))
                df = pd.DataFrame(np.array(fpt).reshape((1, -1)))
                df['index'] = index
                df['path'] = img_path
                pts = pts.append(df, ignore_index=True)
            else:
                if verbos > 0:
                    print('No face/mouth found in %s' % img_path)
        except:
            print(f"error found for {img_path}")

    return pts


def normalize_img(filename, is_vectorize=False, width=96, height=90):
    size = WIDTH, HEIGHT  # (width, height)
    im = Image.open(filename)
    resized_im = im.resize(size, Image.ANTIALIAS)  # resize image
    result = np.array(resized_im)
    #     if is_grey:
    #         im_grey = resized_im.convert('L') # convert the image to *greyscale*
    #         im_array = np.array(im_grey) # convert to np array
    #         result = im_array
    if is_vectorize:
        oned_array = result.reshape(size[0] * size[1])
        result = oned_array
    return result  # np.array(resized_im)#oned_array


def img_generator(data):
    for i in range(len(data)):
        im_info = data.iloc[i]
        index = im_info['index']
        path = im_info['path']
        yield index, path


def get_partial(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def check_creat_dir(save_path):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        print('create folder... %s' % dir_path)
        os.makedirs(dir_path)

WIDTH, HEIGHT = 96, 90 #48*2, 45*2
img = normalize_img(pd_face.iloc[26]['path'], is_vectorize=False, width=WIDTH, height=HEIGHT)
_ = find_pts(img,#imread(pd_face.iloc[23]['path']),
             is_show=True, is_rect=True, verbose=1)