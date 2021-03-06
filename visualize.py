import os
import argparse
import numpy as np
import cv2 as cv
from chainer import serializers
from chainer import Variable
from models.critical_dynamics_model import CriticalDynamicsModel
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


def sample_im(size=256):
    """Return a preprocessed (averaged and resized to VGG) sample image."""
    # mean = np.array([103.939, 116.779, 123.68])
    im = cv.imread('images/double.jpg').astype(np.float32)
    # im -= mean
    im = cv.resize(im, (size, size)).transpose((2, 0, 1))
    im = im[np.newaxis, :, :, :] / 255.0
    return im


def get_activations(model, x, layer, a=None, all_feature=False):
    """Compute the activations for each feature map for the given layer for
    this particular image. Note that the input x should be a mini-batch
    of size one, i.e. a single image.
    """
    if all_feature:
        # a = model.activate_with_attention(Variable(x), Variable(a))  # To 1-indexed
        fm = model(Variable(x), stop_layer=layer)
        a = model.activate_by_feature(fm, layer=layer)
    else:
        a = model.activations(Variable(x), layer=layer)  # To 1-indexed
    a = a.data[0]  # Assume batch with a single image
    a = post_process_activations(a)
    # a = [post_process_activations(_a) for _a in a]
    return a


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


def save_activations_with_attention(model, x, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('Print attention image.')

    a_batch = np.eye(2)[[0]].astype(np.float32)
    activation = get_activations(model, x, layer=5, a=a_batch, all_feature=True)
    save_im(activation, dst_dir + '/attention0.jpg')
    #
    # a_batch = np.eye(2)[[1]].astype(np.float32)
    # activation = get_activations(model, x, layer=5, a=a_batch, all_feature=True)
    # activation = activation[0]
    # im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
    # imgutil.save_im(dst_dir + '/attention1.jpg', im)


def save_im(activation, path):
    activation = activation[0]
    im = np.rollaxis(activation, 0, 3)  # c, h, w -> h, w, c
    imgutil.save_im(path, im)



def save_activations(model, x, layer, dst_root):
    """Save feature map activations for the given image as images on disk."""

    # Create the target directory if it doesn't already exist
    dst_dir = os.path.join(dst_root, 'layer_{}/'.format(layer))
    dst_dir = os.path.dirname(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('Computing activations for layer {}...'.format(layer))
    activations = get_activations(model, x, layer)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='next pred')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    print('Preparing the model...')
    if args.model is not None:
        print(args.model)
        model = CriticalDynamicsModel()
        serializers.load_npz(args.model, model)
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

    if not model._cpu:
        model.to_cpu()
    model.train = False

    # size = Data().insize
    # Visualize each of the 5 convolutional layers in VGG
    # for i in range(len(model.convs)):
    #     save_activations(model, sample_im(size=224), i + 1, outdir)

    # save_activations_with_attention(model, sample_im(size=size), outdir)
    activation = get_activations(model, sample_im(224), layer=5, all_feature=True)
    save_im(activation, outdir + '/all_feature.jpg')
    print('Done')
