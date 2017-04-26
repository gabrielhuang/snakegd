import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_smooth


def load_mnist():
    batch_size = 64
    
    kwargs = {}
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    MAX_DATA = 60000
    
    train_data = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    
    if MAX_DATA is not None:
        train_data.train_data = train_data.train_data[:MAX_DATA]
        train_data.train_labels = train_data.train_labels[:MAX_DATA]
        train_loader_all = torch.utils.data.DataLoader(
            train_data,
            batch_size=MAX_DATA, shuffle=False, **kwargs)    
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=False, **kwargs)
    
    train_loader_stochastic = torch.utils.data.DataLoader(
        train_data,
        batch_size=1, shuffle=False, **kwargs)
    
    
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    
    train_data_tensor, train_labels_tensor = iter(train_loader_all).next()
    train_data_numpy = train_data_tensor.numpy()
    train_labels_numpy = train_labels_tensor.numpy()
    
    return train_data_numpy, train_labels_numpy


def softmax(u, axis=None):
    '''
    Use exp(u_i) / sum(exp(u_j)) = exp(u_i - u_max) / sum(exp(u_j - u_max))
    '''
    u_max = np.max(u, axis=axis, keepdims=True)
    v = u-u_max
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=axis, keepdims=True)

def cross_entropy(true, predicted, axis=None):
    predicted = np.clip(predicted, 1e-15, 1e15)
    return -np.sum(true*np.log(predicted), axis=axis)


class MultiLogisticWeightDecay(object):
    # p(y=i) = exp(w_i^T x) / sum_i exp(w_i^T x) = exp(w_i^T x) / Z
    # loss = - sum_i y_i log p(y=i) = log(Z) - sum_i y_i w_i^T x
    # dloss / dw_i = x (p(y=i) - y_i)
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay
        
    def grad_and_loss(self, w, x, y):
        wx = w.T.dot(x)
        p_y = softmax(wx)  # prediction
        grad_scalar = p_y - y
        grad = np.outer(x, grad_scalar) + self.weight_decay*w/np.prod(w.shape)
        
        loss = cross_entropy(y, p_y) + 0.5*self.weight_decay*np.mean(w*w)
        
        return grad_scalar, grad, loss
    
    def predict_proba(self, w, x):
        return softmax(X.dot(w), axis=1)
    
    
class MultiLogistic(MultiLogisticWeightDecay):
    def __init__(self):
        MultiLogisticWeightDecay.__init__(self, 0.)
    


# question: where to put weight decay?

# SGD
def sgd(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=False):
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    np.random.seed(0)
    # adam part
    momentum = np.zeros_like(w)
    momentum_sqr = np.zeros_like(w)
    beta = 0.9
    beta_sqr = 0.999
    visits = 0
    epsilon = 1e-8

    for t in xrange(nepochs):
        print 'Epoch', t+1
        for i in xrange(len(X)):
    
            i_t = np.random.randint(len(X))
            x = X[i_t]
            y = Y_dummy[i_t]
            visits += 1
            # compute gradient and loss
            grad_scalar, grad, loss = model.grad_and_loss(w, x, y)
            total_grad = grad + float(weight_decay)/w.size*w
            # adam
            if do_adam:
                momentum = beta*momentum + (1-beta)*total_grad
                momentum_sqr = beta_sqr*momentum_sqr + (1-beta_sqr)*total_grad**2
                unbiased = momentum / (1 - beta**visits)
                unbiased_sqr = momentum_sqr / (1 - beta_sqr**visits)
                update = unbiased / (epsilon + np.sqrt(unbiased_sqr))
            else:
                update = total_grad
            # update
            w = w - step_size * update
            total_loss = loss + 0.5*weight_decay*np.mean(w*w)
            # update
            losses.append(total_loss)
    
            if (i+1) % 10000 == 0:
                print 'SGD Epoch {}. {}/{} -- train loss {}'.format(t+1, i+1, len(X), np.mean(losses))
        print '-> train loss {}'.format(np.mean(losses[-len(X):]))
    return w, losses



# SAGA
def saga(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=False):
    global sum_sqr_memories
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    np.random.seed(0)

    memories = np.zeros((X.shape[0], Y_dummy.shape[1]))
    sum_memories = np.zeros_like(w)
    sum_sqr_memories = np.zeros_like(w)
    visited = np.zeros(X.shape[0])

    for t in xrange(nepochs):
        print 'Epoch', t+1
        for i in xrange(len(X)):
    
            i_t = np.random.randint(len(X))
            x = X[i_t]
            y = Y_dummy[i_t]
            visited[i_t] = 1.
                   
            # compute gradient and loss
            grad_scalar, grad, loss = model.grad_and_loss(w, x, y)
            
            # update
            old_grad = np.outer(x, memories[i_t])
            sag_update = grad - old_grad + sum_memories / np.sum(visited)
            memories[i_t] = grad_scalar
            sum_memories = sum_memories + grad - old_grad
            sum_sqr_memories = np.maximum(0., sum_sqr_memories + grad**2 - old_grad**2)
            total_update = (sag_update + float(weight_decay)/w.size*w)
            if do_adam:
                epsilon = 1e-8
                total_update /= (epsilon + np.sqrt(sum_sqr_memories / np.sum(visited)))
            w = w - step_size * total_update
            total_loss = loss + 0.5*weight_decay*np.mean(w*w)
            
            # accumulate
            losses.append(total_loss)
    
            if (i+1) % 10000 == 0:
                print 'SAGA Epoch {}. {}/{} -- train loss {}'.format(t+1, i+1, len(X), np.mean(losses))
        print '-> train loss {}'.format(np.mean(losses[-len(X):]))                
    return w, losses


def get_accuracy(w):
    Y_pred = np.argmax(X.dot(w), axis=1)
    accuracy = np.mean(Y_pred == Y)
    return accuracy
#%% Load MNIST Data
train_data_numpy, train_labels_numpy = load_mnist()


#%% Load smaller digit dataset
from sklearn.datasets import load_digits

digits = load_digits()
train_data_numpy = digits['data']
train_labels_numpy = digits['target']



#%% Transform data, add intercept
skew = True

X = train_data_numpy.reshape((len(train_data_numpy), -1))
if skew:
    # multiply by random scale
    scales = np.random.uniform(size=X.shape[1])
    X = X * scales[np.newaxis, :]
X = np.hstack((np.ones((len(X), 1)), X))
Y = train_labels_numpy
Y_dummy = np.zeros((len(Y), 10))
Y_dummy[np.arange(len(Y)), Y] = 1.



#%% Parameters
model = MultiLogistic()
nepochs = 3
weight_decay = 0.01


#%% SGD
step_size = 0.001
w_sgd, losses_sgd = sgd(X, Y_dummy, step_size, nepochs, weight_decay)
w = w_sgd
losses = losses_sgd

Y_pred = model.predict_proba(w_sgd, X)
print 'train accuracy', get_accuracy(w_sgd)
print 'train loss', cross_entropy(Y_dummy, Y_pred, axis=1).mean()
#%%
step_size = 0.0001
w_sgdadam, losses_sgdadam = sgd(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=True)
w = w_sgdadam
losses = losses_sgdadam

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', cross_entropy(Y_dummy, Y_pred, axis=1).mean()
#%%
step_size = 0.001
w_saga, losses_saga = saga(X, Y_dummy, step_size, nepochs, weight_decay)
w = w_saga
losses = losses_saga

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', cross_entropy(Y_dummy, Y_pred, axis=1).mean()
#%%
step_size = 0.0001
w_sagadam, losses_sagadam = saga(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=True)
w = w_sagadam
losses = losses_sagadam

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', cross_entropy(Y_dummy, Y_pred, axis=1).mean()
#%%
plot_smooth(losses, 'losses', 1000)

#%%
from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(penalty='l2', solver='sag', fit_intercept=False)
clf = LogisticRegression(C=0.1, penalty='l2', solver='sag', fit_intercept=False)
clf.fit(X, Y)
Y_pred = clf.predict_proba(X)
lowest_loss = cross_entropy(Y_dummy, Y_pred, axis=1).mean()
print 'lowest loss', lowest_loss


#%%
N = 4000
semilog = False
#%matplotlib qt
plot_smooth(losses_sgd, 'SGD', N, semilog)
plot_smooth(losses_sgdadam, 'SGD-ADAM', N, semilog)
plot_smooth(losses_saga, 'SAGA', N, semilog)
plot_smooth(losses_sagadam, 'SAGADAM', N, semilog)
plt.legend()
plt.show()

#%% did it learn?
def plot_weights(w):
    for i in xrange(10):
        plt.subplot(4,3,i+1)
        plt.imshow(w[1:, i].T.reshape((28, -1)), cmap='gray')
        
plot_weights(w)

