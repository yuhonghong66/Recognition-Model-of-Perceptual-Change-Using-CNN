import six
import chainer
from chainer import cuda
import numpy as np


class Trainer(object):

    def __init__(self, model, optimizer, loss_function, data, data_feeder, batchsize=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.data = data
        self.data_feeder = data_feeder
        self.batchsize = batchsize
        self.loss_loop = 50
        self.learn_t = 0
        self.sum_loss = 0
        self.xp = np if self.model._cpu else cuda.cupy

    def train(self, n):
        for _ in six.moves.range(n):
            self.learn_t += 1

            # training
            x_batch, t_batch = self.data_feeder(self.batchsize, 3)
            x = chainer.Variable(self.xp.asarray(x_batch))
            t = chainer.Variable(self.xp.asarray(t_batch))

            # model.cleargrads()
            self.model.zerograds()
            loss = self.loss_function(x, t)
            loss.backward()
            self.optimizer.update()
            self.sum_loss += float(loss.data) * len(x.data)

            if self.learn_t % self.loss_loop == 0:
                print('learn_t', self.learn_t)
                print('train mean loss: {}'.format(self.sum_loss / self.batchsize / self.loss_loop))
                self.sum_loss = 0

                # test
                sum_test_loss = 0
                for i in range(self.loss_loop):
                    x_batch, t_batch = self.data_feeder(self.batchsize, 3, test=True)
                    x = chainer.Variable(self.xp.asarray(x_batch), volatile='on')
                    t = chainer.Variable(self.xp.asarray(t_batch), volatile='on')

                    loss = self.loss_function(x, t)
                    sum_test_loss += float(loss.data) * len(x.data)
                print('test mean loss: {}'.format(sum_test_loss / self.batchsize / self.loss_loop))

