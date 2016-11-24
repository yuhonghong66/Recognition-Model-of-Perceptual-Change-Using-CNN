import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from lib.chainer.chainer.functions.pooling import max_pooling_2d
from lib.chainer.chainer.functions.pooling import unpooling_2d


class AttentionModel(chainer.Chain):
    def __init__(self):
        super(AttentionModel, self).__init__(
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 2),
            attention=L.Linear(2, 512)
        )

        self.train = False

    def __call__(self, fm, t=None):
        fm = F.dropout(F.relu(self.fc6(fm)), train=self.train, ratio=0.5)
        fm = F.dropout(F.relu(self.fc7(fm)), train=self.train, ratio=0.5)
        y = self.fc8(fm)

        if self.train:
            self.loss = F.softmax_cross_entropy(y, t)
            self.acc = F.accuracy(y, t)
            return self.loss
        else:
            self.pred = F.softmax(y)
            return self.pred

    def give_attention(self, fm, a):
        attention = self.attention(a)
        shape_attention = attention.data.shape + (1, 1)
        attention = F.reshape(attention, shape_attention)
        shape_tile = (1, 1) + fm.data.shape[2:]
        attention = F.tile(attention, shape_tile)
        fm = fm * attention
        return fm

    def forward_with_attention(self, fm, a, t):
        fm = self.give_attention(fm, a)
        return self(fm, t)
