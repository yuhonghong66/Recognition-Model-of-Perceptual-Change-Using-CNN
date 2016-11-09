from __future__ import print_function
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
try:
    import cPickle as pickle
except:
    import pickle
import chainer
from chainer import cuda

from utils.ML.data import Data
from utils import imgutil

os.environ['PATH'] += ':/usr/local/cuda-7.5/bin'


class Validator(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def validate(self, test=True, attention=False):
        print("start validation!")
        while True:
            if test:
                index = [random.randint(0, data.TEST_N - 1)]
            else:
                index = [random.randint(0, data.N - 1)]

            print("index is " + str(index[0]))

            x_batch, t_batch = self.data.get(index=index, test=test)
            x = chainer.Variable(np.asarray(x_batch))
            if attention:
                a_batch = np.eye(2)[t_batch].astype(np.float32)
                a = chainer.Variable(np.asarray(a_batch))
                t = chainer.Variable(np.asarray(t_batch))
                pred_t = self.model.forward_with_attention(x, a, t).data[0]
            else:
                pred_t = self.model(x).data[0]

            print("Print t_batch")
            print(t_batch[0])
            print("Print prediction")
            print(pred_t)

            img = x_batch[0] * 255
            plt.imshow(img.astype(np.uint8).transpose(1,2,0))
            plt.show()

            print("Press q to quit.")
            key = raw_input()
            if key == 'q':
                return

    def validate_all(self, test=True, attention=False):
        print("validate all picture!")
        score = 0.0
        if test:
            indexes = range(data.TEST_N)
        else:
            indexes= range(data.N)

        for index in indexes:
            x_batch, t_batch = self.data.get(index=[index], test=test)
            x = chainer.Variable(np.asarray(x_batch))
            if attention:
                a_batch = np.eye(2)[t_batch].astype(np.float32)
                a = chainer.Variable(np.asarray(a_batch))
                t = chainer.Variable(np.asarray(t_batch))
                pred_t = self.model.forward_with_attention(x, a, t).data[0]
            else:
                pred_t = self.model(x).data[0]
            if t_batch == np.argmax(pred_t):
                score += 1
        print(score / len(indexes))

    def validate_sample(self, test=True, attention=None):
        print("sample image!")
        x = chainer.Variable(sample_im())
        if attention in [0, 1]:
            print(attention)
            t_batch = [[attention]]
            a_batch = np.eye(2)[t_batch].astype(np.float32)
            a = chainer.Variable(np.asarray(a_batch))
            t = chainer.Variable(np.asarray(t_batch))
            pred_t = self.model.forward_with_attention(x, a, t).data[0]
        else:
            pred_t = self.model(x).data[0]
        print("Print prediction")
        print(pred_t)


def sample_im():
    """Return a preprocessed (averaged and resized to VGG) sample image."""
    # mean = np.array([103.939, 116.779, 123.68])
    im = cv.imread('images/double.jpg').astype(np.float32)
    # im -= mean
    im = cv.resize(im, (128, 128)).transpose((2, 0, 1))
    im = im[np.newaxis, :, :, :] / 255.0
    return im


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='yuragi')
    parser.add_argument('model', type=str)
    parser.add_argument('--test', default=True)
    args = parser.parse_args()

    # make model.
    print(args.model)
    model = pickle.load(open(args.model, 'r'))
    if not model._cpu:
        model.to_cpu()
    model.train = False

    # set hyper param and data.
    data = Data()

    validator = Validator(model, data)
    validator.validate_all(test=args.test)
    validator.validate_sample(test=args.test)
    validator.validate(test=args.test)
    print("Use attention!")
    validator.validate_all(test=args.test, attention=True)
    validator.validate(test=args.test, attention=True)
    validator.validate_sample(test=args.test, attention=0)
    validator.validate_sample(test=args.test, attention=1)


