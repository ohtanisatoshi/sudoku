import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from sklearn.model_selection import train_test_split


class NeuralNet(chainer.Chain):
    def __init__(self):
        super().__init__(
            l1=L.Linear(9, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 100),
            l4=L.Linear(100, 100),
            l5=L.Linear(100, 100),
            l6=L.Linear(100, 100),
            l7=L.Linear(100, 100),
            l8=L.Linear(100, 100),
            l9=L.Linear(100, 100),
            l10=L.Linear(100, 100),
            l11=L.Linear(100, 100),
            l12=L.Linear(100, 100),
            l13=L.Linear(100, 100),
            l14=L.Linear(100, 100),
            l15=L.Linear(100, 100),
            l16=L.Linear(100, 100),
            l17=L.Linear(100, 100),
            l18=L.Linear(100, 100),
            l19=L.Linear(100, 100),
            l20=L.Linear(100, 100),
            l99=L.Linear(100, 9),
        )

        self.y_prod = np.array([362880., 362880., 362880., 362880., 362880., 362880., 362880., 362880., 362880.])
        self.seq_mask = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def __call__(self, x, mask):

        y = self.fwd(x)
        sum_0 = F.sum(y)
        prod_0 = F.prod(y)
        loss = (F.sum(F.square((y * mask) - x)) +
                F.square(sum_0 - 45.))
                #F.log(F.square(prod_0 - 362880.) + 1e-8))
        return loss

    def fwd(self, x):
        h = F.relu(F.local_response_normalization(self.l1(x)))
        h = F.relu(F.local_response_normalization(self.l2(h)))
        h = F.relu(F.local_response_normalization(self.l3(h)))
        h = F.relu(F.local_response_normalization(self.l4(h)))
        h = F.relu(F.local_response_normalization(self.l5(h)))
        h = F.relu(F.local_response_normalization(self.l6(h)))
        h = F.relu(F.local_response_normalization(self.l7(h)))
        h = F.relu(F.local_response_normalization(self.l8(h)))
        h = F.relu(F.local_response_normalization(self.l9(h)))
        h = F.relu(F.local_response_normalization(self.l10(h)))
        h = F.relu(F.local_response_normalization(self.l11(h)))
        h = F.relu(F.local_response_normalization(self.l12(h)))
        h = F.relu(F.local_response_normalization(self.l13(h)))
        h = F.relu(F.local_response_normalization(self.l14(h)))
        h = F.relu(F.local_response_normalization(self.l15(h)))
        h = F.relu(F.local_response_normalization(self.l16(h)))
        h = F.relu(F.local_response_normalization(self.l17(h)))
        h = F.relu(F.local_response_normalization(self.l18(h)))
        h = F.relu(F.local_response_normalization(self.l19(h)))
        h = F.relu(F.local_response_normalization(self.l20(h)))
        h = F.sigmoid(F.local_response_normalization(self.l99(h)))
        h = (h * 8) + 1
        return h

def main():
    q = np.array([[1,2,0,5,0,3,0,4,0]]).astype(np.float32)
    q_mask = q.astype(np.bool).astype(np.float32)
    a = np.array([[6,1,9,7,3,2,5,4,8,
                   8,5,3,9,4,1,6,2,7,
                   4,2,7,6,5,8,1,9,3,
                   5,7,1,2,9,4,3,8,6,
                   2,3,8,5,7,6,4,1,9,
                   9,6,4,1,8,3,7,5,2,
                   3,9,5,4,2,7,8,6,1,
                   1,8,2,3,6,5,9,7,4,
                   7,4,6,8,1,9,2,3,5
                   ]]).astype(np.float32)

    model = NeuralNet()

    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    epoch = 3000
    for i in range(epoch):
        model.cleargrads()
        loss = model(q, q_mask)
        loss.backward()
        optimizer.update()

        if i % 10 == 0:
            print('Epoch {} loss(train) = {:.6f}'.format(i+1, float(loss.data)))
    y = model.fwd(q)
    for i in range(9):
        print('{:.0f},'.format(np.round(float(y.data[0][i]))), end="")


if __name__ == '__main__':
    main()
