#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "chen"

import time
import cv2 as cv
from PIL import Image
import numpy as np

from sklearn.preprocessing import normalize
from keras.models import load_model
from skimage.io import imshow, imread, imsave

import auxiliary

import csv



if __name__ == '__main__':

    md = auxiliary.Mouth_Decector()

    model = load_model('./model/model_smile_rec_by_nn.h5')

    cap = cv.VideoCapture(0)

    print('start capature video')

    fps = 60
    width = 48 # 28
    height = 28 # 10

    loop_time = []

    # with open('TimeRecord_MultiFace_use_yield', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

    loop = 1

    while True:
        start_time = time.time()

        _, frame = cap.read()
        img_org = cv.resize(frame, (640, 360))
        # mouths = md.find_mouths(img_org, is_square_face=True, is_square_mouth=True)

        # if (len(mouths) > 0):
        count = 1

        loop_start = time.time()
        # loop_time.append('Loop ' + str(loop) + ' start at ' + str(loop_start))

        for mouth, fw in md.find_mouths(img_org, is_square_face=True, is_square_mouth=True):
            this_start = time.time()
            # loop_time.append('Img '+str(count)+' start process at '+str(this_start))


            x, y, w, h = mouth
            mouth_img = md.get_partial(img_org, x, y, w, h)

            # imsave(mouth_img, './my_face')
            mouth_vec = md.normalize_img(mouth_img, is_vectorize=False, width=width, height=height)
            data_img_vec_norm = mouth_vec / 255 # normalize(mouth_vec, norm='l2', axis=1)
            data_img_vec_norm = data_img_vec_norm.reshape(-1, height, width, 1)

            result = model.predict(data_img_vec_norm)

            rate = 0

            # if w<100:
            #     rate = 0.0001* pow(2, w/10)-1  #不够
            if fw < 300:
                rate = 0.621*np.log(fw / 60.) - 1.  #还可以再多补一点

            is_smile = result[0][1] - result[0][0] > rate

            if is_smile:
                cv.putText(img_org,
                               'smile',
                               (x, y - 10),
                               cv.FONT_HERSHEY_DUPLEX,
                               2,
                               (97, 50, 205)) # 92, 92, 205
            else:
                cv.putText(img_org,
                           'not smile',
                           (x, y - 10),
                           cv.FONT_HERSHEY_DUPLEX,
                           1.5,
                           (28, 28, 28))

            print(f"{mouth}, is_smile: {is_smile}, smile_prop: {result[0][1]}")

            this_end = time.time()

            # loop_time.append('Img'+str(count)+' end process at '+str(this_end))
            # loop_time.append('Img '+str(count)+' process time: ' + str(round(this_end - this_start,4))+ ' in Loop '+str(loop))

            count += 1

            # loop_time.append(thisloop)

            # time.sleep(1.0 / fps)
        cv.imshow("looking", img_org)

        loop_end = time.time()
        # loop_time.append('Loop ' + str(loop) + ' start at ' + str(loop_end))
        # loop_time.append('Loop ' + str(loop) + ' time: ' + str(round(loop_end - loop_start, 4)))

        if (cv.waitKey(10) & 0xFF == ord('q')):
            cap.release()
            # cv.destroyallwindows()
            break

        loop += 1

    for row in loop_time:
        print(row)
    # print(loop_time)

    # with open('TimeRecord_MultiFace', 'w') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     for row in loop_time:
    #         wr.writerow([row])

