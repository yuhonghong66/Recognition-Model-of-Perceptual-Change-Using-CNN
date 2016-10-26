# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
try:
   import cPickle as pickle
except:
   import pickle

from scraper import scrape_image
from bing import Bing
from detector import detect


def collect_data():
    key = "TIwk7p7nC7HlKijRb5Z42IHx0S2+MKHqAS0BNIOdKqM"
    name_list = ['ヒラリー・クリントン', 'ビル・クリントン']
    bing = Bing(key)
    save_dir = './raw_image/'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for name in name_list:
        save_dir = './raw_image/' + name + '/'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        results = bing.web_search(name, 200, ["MediaUrl"])

        for num, result in enumerate(results):
            try:
                scrape_image(result['MediaUrl'], save_dir + str(num) + '.jpg')
            except Exception as e:
                print(e)
                continue


def detect_face():
    name_list = ['ヒラリー・クリントン', 'ビル・クリントン']
    processed_dir = './processed_image/'
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    for name in name_list:
        path = './raw_image/' + name + '/'
        image_list = os.listdir(path)
        image_list = filter(lambda x: x[0] != '.', image_list)

        for image_name in image_list:
            img = cv2.imread(path+image_name)

            try:
                img = detect(img)
            except Exception as e:
                print(e)
                print(name + ':' + image_name)
                continue

            save_path = processed_dir + name + '/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imwrite(save_path+image_name, img)


def make_txtfile():
    name_list = ['ヒラリー・クリントン', 'ビル・クリントン']
    processed_dir = './processed_image/'

    with open('label.txt', 'w') as f:
        for num, name in enumerate(name_list):
            f.write(name + ' ' + str(num) + '\n')

    with open('data.txt', 'w') as f:
        for num, name in enumerate(name_list):
            image_path = processed_dir + name + '/'
            image_list = os.listdir(image_path)
            image_list = filter(lambda x: x[0] != '.', image_list)

            for path in image_list:
                f.write(image_path + path + ' ' + str(num) + '\n')


def load_image_list(path):
    tuples = []
    with open(path, 'r') as f:
        for line in f:
            pair = line.strip().split()
            tuples.append(pair)
    return tuples


class Data(object):
    def __init__(self):
        data = load_image_list('data.txt')
        n_data = len(data)
        self.insize = 128
        self.train = data[:n_data / 10 * -1]
        self.test = data[n_data / 10 * -1:]
        self.N = len(self.train)
        self.TEST_N = len(self.test)

    def read_image(self, path):
        # Data loading routine
        resize_img = Image.open(path).resize((self.insize, self.insize))
        img = np.asarray(resize_img).transpose(2, 0, 1)
        img = img.astype(np.float32)
        img /= 255

        if random.randint(0, 1) == 0:
            return img[:, :, ::-1]
        else:
            return img

    def get(self, index, test=False):
        # send list or tuple data (not numpy)
        x_batch = np.ndarray(
            (len(index), 3, self.insize, self.insize), dtype=np.float32)
        t_batch = np.ndarray(
            (len(index),), dtype=np.int32)

        if test:
            data_set = self.test
        else:
            data_set = self.train

        for k, idx in enumerate(index):
            path = data_set[idx][0]
            target = data_set[idx][1]
            img = self.read_image(path)
            #TODO: tolist may be very slow!
            x_batch[k] = img
            t_batch[k] = target

        return x_batch, t_batch


if __name__ == '__main__':
    # collect_data()
    # detect_face()
    make_txtfile()
    pass
