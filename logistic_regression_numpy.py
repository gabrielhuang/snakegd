import numpy as np
from tqdm import tqdm
from collections import deque

np.random.seed(1234)

def sigmoid(x):
    s = 1.0 /(1.0 + np.exp(-x))
    return s

class SGD(object):
    def __init__(self, lr=0.1):
        self.lr = lr

    def train(self, model, x, y, i):
        loss = model.forward(x[i], y[i])
        grad = model.backward(x[i], y[i])
        update = -self.lr*grad
        model.update(update)

        return loss


class SVRG(object):
    def __init__(self, lr=0.1, update_frequency=100, auto_epoch=True, batching=True):
        self.lr = lr
        self.update_frequency = update_frequency
        self.auto_epoch = auto_epoch
        self.update_flag = True
        self.previous_diff = 0.
        self.count = 0
        self.s = 0

    def train(self, model, x, y, i):
        if (self.auto_epoch and self.update_flag) or (not self.auto_epoch and self.count%self.update_frequency == 0):
            self.w_copy = model.get_w()
            self.mu = 0.
            if batching:
                batch_size = min(2**self.s, len(x))
            else:
                batch_size = len(x)
            idx = np.random.permutation(len(data))[:batch_size]
            for j in idx:
                self.mu += model.backward(x[j],y[j], self.w_copy)
            self.mu /= len(data)

            self.diff = []
            self.update_flag = False
            self.s += 1

        loss = model.forward(x[i], y[i])
        grad = model.backward(x[i], y[i])
        grad_copy = model.backward(x[i], y[i], self.w_copy)
        d = grad - grad_copy
        self.diff.append(np.sum((d)**2))
        update = -self.lr*( d + self.mu)
        model.update(update)

        if (np.mean(self.diff[-self.update_frequency:]) > self.previous_diff/2.) and (len(self.diff) >= self.update_frequency):
            self.update_flag = True
            self.previous_diff = np.mean(self.diff[-self.update_frequency:])
        self.count += 1

        return loss

class LogisticRegression(object):
    def  __init__(self, n_features, l2=0.):
        self.w = np.ones(n_features)
        self.l2 = l2

    def forward(self, x, y, w=None):
        if w is None:
            w = self.w
        mu = sigmoid(np.dot(x, w))
        f = - y*np.log(mu) - (1-y)*np.log(1-mu) + 1/2*self.l2*np.sum(w[:-1]**2)
        return f

    def backward(self, x, y, w=None):
        if w is None:
            w = self.w
        mu = sigmoid(np.dot(x, w))
        return (mu-y)*x

    def update(self, update):
        self.w += update

    def get_w(self):
        return self.w.copy()

    def set_w(self, w):
        self.w = w.copy()

data = np.loadtxt('iris.txt', delimiter=',')
np.random.shuffle(data)
data = data[data[:,-1]<3,:]
data[data[:,-1]==2,-1] = 0

x , y = np.ones(data.shape), data[:,-1]
data = data[:,:-1]
Umin = np.min(data, axis=0)
Umax = np.max(data, axis=0)
data = (data- Umin)/(Umax-Umin)
x[:,:-1] = data


n_epochs = 100
update_frequency = len(data)/4
L = 0.25*np.max(np.sum(x**2, axis=1))
lr = 1.0/L
update_flag = True
previous_diff = 0

model = LogisticRegression(data.shape[1]+1)
#optimizer = SGD(lr=0.1)
optimizer = SVRG(lr, update_frequency=update_frequency, auto_epoch=True)

for epoch in range(n_epochs):
    np.random.shuffle(data)
    loss = 0.
    for i in range(len(data)):
        loss += optimizer.train(model, x, y, i)

    print loss/len(data)
