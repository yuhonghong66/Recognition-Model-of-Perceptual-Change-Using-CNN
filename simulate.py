import matplotlib.pyplot as plt
import numpy as np


def sample_simulate():
    print("simulate!")
    result = []

    # double
    no_attention = 0.57152784
    attention0 = 0.99874961
    attention1 = 0.01055572

    # x = chainer.Variable(sample_im(size=self.data.insize))
    # fm = self.VGG(x, stop_layer=5)
    # pred_t = self.model(fm).data[0]

    if no_attention > np.random.rand():
        threshold = attention0
    else:
        threshold = attention1

    for _ in range(10000):
        if threshold > np.random.rand():
            threshold = attention0
            perception = 0
        else:
            threshold = attention1
            perception = 1

        # t_batch = [[attention]]
        # a_batch = np.eye(2)[t_batch].astype(np.float32)
        # a = chainer.Variable(np.asarray(a_batch))

        # fm = self.VGG(x, stop_layer=5)
        # pred_t = self.model.forward_with_attention(fm, a).data[0]

        result.append(perception)

    print("simulated!")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(result)+1), result)
    plt.ylim([-0.3, 1.3])
    plt.tick_params(labelleft='off')
    plt.savefig('simulate_double.jpg')

    print(result)

if __name__ == '__main__':
    sample_simulate()

