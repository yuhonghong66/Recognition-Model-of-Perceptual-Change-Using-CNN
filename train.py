from __future__ import print_function
import argparse
import numpy as np
import six
import os
try:
    import cPickle as pickle
except:
    import pickle
import chainer
from chainer import cuda
from chainer import optimizers
from utils.ML.data import Data
from models.critical_dynamics_model import CriticalDynamicsModel
from utils.prepare_output_dir import prepare_output_dir
from utils.plot_scores import plot_scores

os.environ['PATH'] += ':/usr/local/cuda-7.5/bin'

# parse args
parser = argparse.ArgumentParser(description='next pred')
parser.add_argument('--gpu', '-g', default=-1, type=int)
parser.add_argument('--batchsize', '-b', default=1, type=int)
parser.add_argument('--epoch', '-e', default=30, type=int)
args = parser.parse_args()

# make model.
model = CriticalDynamicsModel()
model.train = True
optimizer = optimizers.Adam()
# optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# set hyper param and data.
batchsize = args.batchsize
n_epoch = args.epoch

data = Data()
N = data.N
TEST_N = data.TEST_N

log_dir = prepare_output_dir(args)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0       # total loss
    for i in six.moves.range(0, N, batchsize):
        x_batch, t_batch = data.get(perm[i: i+batchsize])
        x = chainer.Variable(xp.asarray(x_batch))
        t = chainer.Variable(xp.asarray(t_batch))

        # model.cleargrads()
        model.zerograds()
        loss = model(x, t)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x.data)

    print('train mean loss: {}'.format(sum_loss / N))

    # test
    x_batch, t_batch = data.get(range(TEST_N), test=True)
    x = chainer.Variable(xp.asarray(x_batch))
    t = chainer.Variable(xp.asarray(t_batch))

    loss = model(x, t)
    print('test mean loss: {}'.format(float(loss.data)))
    if not os.path.exists(log_dir+'/loss.txt'):
        with open(log_dir+'/loss.txt', 'a') as f:
            f.write('learn_t,train_loss,test_loss\n')
    with open(log_dir+'/loss.txt', 'a') as f:
        f.write(str(epoch) + ',' + str(sum_loss / data.N) + ',' + str(loss.data) + '\n')

# save model.
pickle.dump(model, open(log_dir + '/model.pkl', 'wb'), protocol=2)
pickle.dump(optimizer, open(log_dir + '/optimizer.pkl', 'wb'), protocol=2)
plot_scores(log_dir+'/loss.txt')
