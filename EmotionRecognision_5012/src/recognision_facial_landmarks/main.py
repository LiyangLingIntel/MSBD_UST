#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "chen"

import time
import cv2 as cv
from PIL import Image
import numpy as np
import dlib
from time import time

from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
# from keras.models import load_model
from skimage.io import imshow, imread, imsave
import auxiliary


def get_feature_pts(img, face, predictor, is_tag=False, is_normalize=True):

    pts = []

    shape = predictor(img, face)
    face_width = face.right() - face.left()
    face_height = face.bottom() - face.top()

    for i in range(36, 68):
        if is_tag:
            img = cv.circle(img.copy(), (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), 1)
            cv.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
        x = shape.part(i).x
        y = shape.part(i).y

        if is_normalize:
            # Should be changed because the total image should base on face, not origin pic
            x = (shape.part(i).x - face.left()) / face_width
            y = (shape.part(i).y - face.top()) / face_height
        #             pts.append([x, y])
        pts.extend([x, y])

    return pts


if __name__ == '__main__':

    md = auxiliary.Mouth_Decector()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../01_dataset_GENKI_4K/model/shape_predictor_68_face_landmarks.dat')

    # model = joblib.load('../01_dataset_GENKI_4K/model/68pts_svm.joblib')
    model = joblib.load('../01_dataset_GENKI_4K/model/32pts_svm.joblib')


    cv.namedWindow('hello_smile')

    vc = cv.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    cap = cv.VideoCapture(0)

    print('start capature video')

    fps = 120
    width = 48 # 28
    height = 28 # 10

    while True:
        # start_time = time()

        _, img_org = cap.read()

        cv.resize(img_org, (640, 360))

        # mouth = md.find_mouth(img_org, is_square_face=True, is_square_mouth=True)
        faces_detected = detector(img_org, 1)
        print('Number of faces detected: {}'.format(len(faces_detected)))

        s1 = time()
        for i, face in enumerate(faces_detected):

            img_org = cv.rectangle(img_org.copy(), (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
                i, face.left(), face.top(), face.right(), face.bottom()))

            s2 = time()
            feature_pts = get_feature_pts(img_org, face, predictor, is_tag=True)
            print(f'Time-cost for 68 point: {time()-s2}')

            feature_pts = np.array(feature_pts).reshape(1, -1)
            s3 = time()
            prediction = model.predict(feature_pts)
            print(f'Time-cost for prediction: {time()-s3}')

            is_smile = True if prediction > 0 else False

            if is_smile:
                cv.putText(img_org,
                           'smile',
                           (face.left()+10, face.top() - 10),
                           cv.FONT_HERSHEY_DUPLEX,
                           2,
                           (97, 50, 205)) # 92, 92, 205
            else:
                cv.putText(img_org,
                           'not smile',
                           (face.left(), face.top() - 10),
                           cv.FONT_HERSHEY_DUPLEX,
                           1.5,
                           (0, 0, 0))

            print(f"{i}, is_smile: {is_smile}, smile_prop: {1}")

        print(f'Time-cost for one face: {time()-s1}')

        cv.imshow("looking", img_org)

        # time.sleep(1.0 / fps)

        if (cv.waitKey(10) & 0xFF == ord('q')):
            cap.release()
            # cv.destroyallwindows()
            break
