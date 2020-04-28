from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.models import load_model

IMAGE_MAX_X = 512
IMAGE_MAX_Y = 512

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class TwoPoint:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class CrackManager:
    def __init__(self, modelPath, configPath, classificationPath = ''):
        self.modelPath = modelPath
        self.configPath = configPath

        print('Model loading...')

        self.classification_model = load_model(classificationPath)

        with open(self.configPath, 'rb') as f_in:
            self.C = pickle.load(f_in)

        if self.C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif self.C.network == 'vgg':
            import keras_frcnn.vgg as nn

        self.class_mapping = self.C.class_mapping

        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}

        self.C.num_rois = 4

        if self.C.network == 'resnet50':
            num_features = 1024
        elif self.C.network == 'vgg':
            num_features = 512

        if K.image_data_format() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping),
                                   trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)

        self.model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(self.C.model_path))
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        self.model_classifier.load_weights(self.C.model_path, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def predict(self, img_path):
        all_imgs = []
        classes = {}
        bbox_threshold = 0.8

        st = time.time()

        img = cv2.imread(img_path)

        X, ratio = format_img(img, self.C)
        print(K.image_data_format())
        if K.image_data_format() == 'tf' or K.image_data_format() == 'channels_last':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_data_format(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois * jk:self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):

                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        candidates = [] # 모든 객체 후보 영역이 담길 list

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                        overlap_thresh=0.6)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                real_points = TwoPoint(real_x1 + 10, real_y1 + 10, real_x2 - 10, real_y2 - 10)

                candidates.append(real_points)

                cv2.rectangle(img, (real_x1 + 10, real_y1 + 10), (real_x2 - 10, real_y2 - 10), (255, 0, 0), 2)
                # cv2.rectangle(img,(x1, y1), (x2, y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 3)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0], textOrg[1]), (textOrg[0] + retval[0], textOrg[1] - retval[1]),
                              (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)

                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        min_x1 = 10
        min_y1 = 10
        max_x2 = 0
        max_y2 = 0

        for c in candidates: # 허용된 범위를 넘어가는 bbox에 대한 최대 좌표 수정
            if c.x1 <= 0:
                c.x1 = 10

            if c.y1 <= 0:
                c.y1 = 10

            if c.x2 >= IMAGE_MAX_X:
                c.x2 = IMAGE_MAX_X - 10

            if c.y2 >= IMAGE_MAX_Y:
                c.y2 = IMAGE_MAX_Y - 10

            if min_x1 > c.x1: # 객체들의 영역을 하나의 영역으로 합치는 과정
                min_x1 = c.x1

            if min_y1 > c.y1:
                min_y1 = c.y1

            if max_x2 < c.x2:
                max_x2 = c.x2

            if max_y2 < c.y2:
                max_y2 = c.y2 # 각 객체의 최소점, 최대점을 계산하여 하나로 합침

        cv2.rectangle(img, (min_x1, min_y1), (max_x2, max_y2), (255, 0, 255), 2)

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        cv2.imshow('img', img)
        cv2.waitKey(0)

c = CrackManager('./model_frcnn_hdf5', './config.pickle')
c.predict('./testData/1.60233859-cracked-concrete-texture-closeup-background.jpg')
c.predict('./testData/4.concrete-712.jpg')
c.predict('./testData/1.guide-to-cracks-in-brick-walls.jpg')
c.predict('./testData/2.14511448-brick-wall-with-big-diagonal-crack.jpg')
c.predict('./testData/2.3509673328_9b7ed82212_b-5705a84d3df78c7d9e953054.jpg')