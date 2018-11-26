import chainer
import chainer.links as L
import chainer.functions as F


class LeNet(chainer.Chain):
    """
    """

    def __init__(self, n_class, binary=True):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 6, 5, stride=1)
            self.conv2 = L.Convolution2D(None, 16, 5, stride=1)
            self.fc3 = L.Linear(None, 120)
            self.fc4 = L.Linear(None, 64)
            self.fc5 = L.Linear(None, n_class)
        self.binary = binary

    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        if self.binary:
            h = h[:, 0]
        return h
