import os
import argparse
import numpy as np
import cv2 as cv
from chainer import serializers
from chainer import Variable
from models.attention_model import AttentionModel
from models.VGG import VGG
from utils import imgutil
try:
    import cPickle as pickle
except:
    import pickle
from utils.ML.data import Data

"""
TODO
- Speed up the unpooling loop with indexes loop
- Suport GPU
"""

def save_im(activation, path):
    activation = activation[0]
    im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
    imgutil.save_im(path, im)


def sample_im(size=256):
    """Return a preprocessed (averaged and resized to VGG) sample image."""
    # mean = np.array([103.939, 116.779, 123.68])
    im = cv.imread('images/bill.jpg').astype(np.float32)
    # im -= mean
    im = cv.resize(im, (size, size)).transpose((2, 0, 1))
    im = im[np.newaxis, :, :, :] / 255.0
    return im


def post_process_activations(a):
    # Center at 0 with std 0.1
    a -= a.mean()
    a /= (a.std() + 1e-5)
    a *= 0.1

    # Clip to [0, 1]
    a += 0.5
    a = np.clip(a, 0, 1)

    # To RGB
    a *= 255
    # a = a.transpose((1, 2, 0))
    a = np.clip(a, 0, 255).astype('uint8')

    return a


def post_process_activations_simple(a):
    a -= a.min()
    if a.max() > 0:
        a *= 255.0 / a.max()
    else:
        a *= 255.0
    return a


class Visualizer(object):
    def __init__(self, model, attention_model=None):
        self.model = model
        self.attention_model = attention_model

    def get_activations(self, x, layer, a=None, all_feature=False):
        if all_feature:
            fm = self.model(x, stop_layer=layer)
            if a is not None:
                fm = self.attention_model.give_attention(fm, a)
            activations = self.model.activate_by_feature(fm, layer=layer)
        else:
            activations = self.model.activations(x, layer=layer)

        activations = activations.data[0]
        activations = post_process_activations(activations)
        # activations = post_process_activations_simple(activations)
        # activations = [post_process_activations(_a) for _a in activations]
        # activations = [post_process_activations_simple(_a) for _a in activations]
        return activations

    def save_activations(self, x, layer, dst_root):
        # Create the target directory if it doesn't already exist
        dst_dir = os.path.join(dst_root, 'layer_{}/'.format(layer))
        dst_dir = os.path.dirname(dst_dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        activations = self.get_activations(x, layer)

        # Save each activation as its own image to later tile them all into
        # a single image for a better overview
        filename_len = len(str(len(activations)))
        for i, activation in enumerate(activations):
            im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
            filename = os.path.join(dst_dir,
                                    '{num:0{width}}.jpg'  # Pad with zeros
                                    .format(num=i, width=filename_len))

            print('Saving image {}...'.format(filename))
            imgutil.save_im(filename, im)

            tiled_filename = os.path.join(dst_root, 'layer_{}.jpg'.format(layer))
        print('Saving image {}...'.format(filename))
        imgutil.tile_ims(tiled_filename, dst_dir)

    def print_feature_map(self, x, dst_dir, layer=5, a=None):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if a is not None:
            for idx in a:
                attention = Variable(np.eye(2)[[idx]].astype(np.float32))
                fm = self.model(x, stop_layer=layer)
                fm = self.attention_model.give_attention(fm, attention)
                activations = self.model.activate_by_feature(fm, layer=layer)
                activations = activations.data[0]
                activations = post_process_activations(activations)
                # activations = post_process_activations_simple(activations)
                # activations = [post_process_activations_simple(_a) for _a in activations]
                save_im(activations, outdir + '/all_feature' + str(idx) + '.jpg')
        else:
            fm = self.model(x, stop_layer=layer)
            activations = self.model.activate_by_feature(fm, layer=layer)
            activations = activations.data[0]
            activations = post_process_activations(activations)
            # activations = post_process_activations_simple(activations)
            # activations = [post_process_activations_simple(_a) for _a in activations]
            save_im(activations, outdir + '/all_feature.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='next pred')
    parser.add_argument('model', type=str, default=None)
    args = parser.parse_args()

    print('Preparing the model...')
    if args.model is not None:
        print(args.model)
        model = VGG()
        serializers.load_hdf5('VGG.model', model)
        attention_model = AttentionModel()
        serializers.load_npz(args.model, attention_model)
        attention_model.train = False
        model_name = args.model.split('/')[-1]
        outdir = args.model.rstrip(model_name) + 'activations'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        model = VGG()
        serializers.load_hdf5('VGG.model', model)
        outdir = 'activations'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    model.train = False

    visualizer = Visualizer(model, attention_model=attention_model)

    size = 224
    x = Variable(sample_im(size=224))
    # Visualize each of the 5 convolutional layers in VGG
    # for i in range(len(model.convs)):
    #     visualizer.save_activations(x, i + 1, outdir)
    visualizer.save_activations(x, 1, outdir)

    # visualizer.print_feature_map(x, dst_dir=outdir, a=[0, 1])
    # visualizer.print_feature_map(x, dst_dir=outdir)

    print('Done')
