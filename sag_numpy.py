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
    
    def loss(self, w, X, Y_true, Y_pred):
        return cross_entropy(Y_true, Y_pred, axis=1).mean() + 0.5*self.weight_decay*np.mean(w*w)
    
    def predict_proba(self, w, X):
        return softmax(X.dot(w), axis=1)
    
    
class MultiLogistic(MultiLogisticWeightDecay):
    def __init__(self):
        MultiLogisticWeightDecay.__init__(self, 0.)
    


# question: where to put weight decay?
class Adam(object):
    def __init__(self, param_shape, beta=0.9, beta_sqr=0.999, epsilon=1e-8):
        self.param_shape = param_shape
        self.beta = beta
        self.beta_sqr = beta_sqr
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        self.visits = 0
        self.momentum = np.zeros(self.param_shape)
        self.momentum_sqr = np.zeros(self.param_shape)
        
    def set_grad(self, grad):
        self.visits += 1
        self.momentum = self.beta*self.momentum + (1-self.beta)*grad
        self.momentum_sqr = self.beta_sqr*self.momentum_sqr + (1-self.beta_sqr)*grad**2
        self.unbiased = self.momentum / (1 - self.beta**self.visits)
        self.unbiased_sqr = self.momentum_sqr / (1 - self.beta_sqr**self.visits)
        self.denom = (self.epsilon + np.sqrt(self.unbiased_sqr))
        self.update = self.unbiased / self.denom    

    def get_update(self):
        return self.update

        
        
# SGD
def sgd(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=False, debug=False):
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    if do_adam:
        adam = Adam(w.shape)

    for t in xrange(nepochs):
        print 'Epoch', t+1
        for i in xrange(len(X)):
    
            i_t = np.random.randint(len(X))
            x = X[i_t]
            y = Y_dummy[i_t]
            # compute gradient and loss
            grad_scalar, grad, loss = model.grad_and_loss(w, x, y)
            total_grad = grad + float(weight_decay)/w.size*w
            # adam
            if do_adam:
                adam.set_grad(grad)
                update = adam.get_update()
                if debug:
                    print 'epoch {} example {}'.format(t+1, i+1)
                    print 'update.max()', update.max()
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
def saga(X, Y_dummy, step_size, nepochs, weight_decay, do_adam=False, do_sagadam=False, debug=False):
    global sum_sqr_memories
    global w
    global adam
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    np.random.seed(0)

    memories = np.zeros((X.shape[0], Y_dummy.shape[1]))
    sum_memories = np.zeros_like(w)
    sum_sqr_memories = np.zeros_like(w)
    visited = np.zeros(X.shape[0])
    
    if do_adam:
        adam = Adam(w.shape)
    
    assert(not (do_adam and do_sagadam))

    for t in xrange(nepochs):
        print 'Epoch', t+1
        for i in xrange(len(X)):
    
            i_t = np.random.randint(len(X))
            x = X[i_t]
            y = Y_dummy[i_t]
            visited[i_t] = 1.
                   
            # compute gradient and loss
            grad_scalar, grad, loss = model.grad_and_loss(w, x, y)
            
            # SAG update
            old_grad = np.outer(x, memories[i_t])
            sag_update = grad - old_grad + sum_memories / np.sum(visited)
            memories[i_t] = grad_scalar
            sum_memories = sum_memories + grad - old_grad
            sum_sqr_memories = np.maximum(0., sum_sqr_memories + grad**2 - old_grad**2)            

            if do_sagadam:
                epsilon = 1e-8
                # problems when visited is small, need to bias
                denom = (epsilon + np.sqrt(sum_sqr_memories / np.sum(visited)))
                #visited_ratio = (np.sum(visited)/float(len(X)))
                #print 'visited ratio', visited_ratio
                #denom = (epsilon 
                #     + (1-visited_ratio)*np.sqrt(np.mean(sum_sqr_memories) / np.sum(visited))
                #     + visited_ratio*(np.sqrt(sum_sqr_memories / np.sum(visited))))
                    
                # cannot really do sagadam on the L2 part
                total_update = sag_update/denom + float(weight_decay)/w.size*w
            elif do_adam:
                # do adam on L2 part
                adam.set_grad(sag_update + float(weight_decay)/w.size*w)
                total_update = adam.get_update()
            else:  # normal
                total_update = sag_update + float(weight_decay)/w.size*w

            if debug:
                print 'epoch {} example {}'.format(t+1, i+1)
                print 'total_update.max()', total_update.max()                


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


def skew(X, power=5.):
    # multiply by random scale
    scales = np.random.uniform(size=X.shape[1]) ** power
    X = X * scales[np.newaxis, :]
    return X

#%% Load Data
dataset = 'mnist'
#dataset = 'digits'
do_skew = False

if dataset == 'mnist':
    train_data_numpy, train_labels_numpy = load_mnist()
elif dataset == 'digits':
    # Load smaller digit dataset
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    train_data_numpy = digits['data']
    train_labels_numpy = digits['target']
    train_data_numpy /= train_data_numpy.max()
else:
    raise Exception('Unknown dataset')

# to X, Y
X = train_data_numpy.reshape((len(train_data_numpy), -1))
Y = train_labels_numpy
Y_dummy = np.zeros((len(Y), 10))
Y_dummy[np.arange(len(Y)), Y] = 1.

# skew data
if do_skew:
    X = skew(X)
        
# add intercept
X = np.hstack((np.ones((len(X), 1)), X))


        
#%% Parameters
model = MultiLogistic()
params = {
        'nepochs': 20,
        'weight_decay': 0.01,
}


if dataset == 'mnist':
    params.update({
        'smooth': 60000,
        'nepochs': 3,
        'sgd': 0.001,
        'adam': 0.0001,
        'saga': 0.001,
        'saga.adam': 0.0001,
        'sagadam': 0.0005,
    })
elif dataset == 'digits':
    params.update({
        'smooth': 4000,
        'nepochs': 40,
        'sgd': 0.05,
        'adam': 0.001,
        'saga': 0.1,
        'saga.adam': 0.01,
        'sagadam': 0.005,
    })
else:
    raise Exception('Unknown dataset')

# raise on NAN
np.seterr(all='warn')
np.seterr(over='raise', divide='raise')

#%% SGD
w_sgd, losses_sgd = sgd(X, Y_dummy, params['sgd'], params['nepochs'], params['weight_decay'])
w = w_sgd
losses = losses_sgd

Y_pred = model.predict_proba(w_sgd, X)
print 'train accuracy', get_accuracy(w_sgd)
print 'train loss', model.loss(w, X, Y_dummy, Y_pred)
#%%
w_sgdadam, losses_sgdadam = sgd(X, Y_dummy, params['adam'], params['nepochs'], params['weight_decay'], do_adam=True)
w = w_sgdadam
losses = losses_sgdadam

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', model.loss(w, X, Y_dummy, Y_pred)
#%%
w_saga, losses_saga = saga(X, Y_dummy, params['saga'], params['nepochs'], params['weight_decay'])
w = w_saga
losses = losses_saga

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', model.loss(w, X, Y_dummy, Y_pred)

#%%
w_saga_adam, losses_saga_adam = saga(X, Y_dummy, params['saga.adam'], params['nepochs'], params['weight_decay'], do_adam=True)
w = w_saga_adam
losses = losses_saga_adam

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_saga_adam)
print 'train loss', model.loss(w, X, Y_dummy, Y_pred)

#%%
w_sagadam, losses_sagadam = saga(X, Y_dummy, params['sagadam'], params['nepochs'], params['weight_decay'], do_adam=True, debug=False)
w = w_sagadam
losses = losses_sagadam

Y_pred = model.predict_proba(w, X)
print 'train accuracy', get_accuracy(w_sgdadam)
print 'train loss', model.loss(w, X, Y_dummy, Y_pred)


#%%
from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(penalty='l2', solver='sag', fit_intercept=False)
clf = LogisticRegression(C=0.01, penalty='l2', fit_intercept=False)
clf.fit(X, Y)
Y_pred = clf.predict_proba(X)
lowest_loss = model.loss(clf.coef_, X, Y_dummy, Y_pred)
print 'lowest loss', lowest_loss


#%%
%matplotlib qt
import utils
reload(utils)
from utils import plot_smooth

# todo: also plot with almost no smoothing to show stability

#for i, semilog in enumerate((True, False)):
for i, semilog in enumerate((False,)):
    plt.figure(1+i)
    N = params['smooth']
    plot_smooth(losses_sgd, 'SGD', N, semilog)
    plot_smooth(losses_sgdadam, 'SGD-ADAM', N, semilog)
    plot_smooth(losses_saga, 'SAGA', N, semilog)
    plot_smooth(losses_saga_adam, 'SAGA-ADAM', N, semilog)
    plot_smooth(losses_sagadam, 'SAGADAM', N, semilog)
    plt.legend()
plt.show()

#%% did it learn?
def plot_weights(w):
    for i in xrange(10):
        plt.subplot(4,3,i+1)
        plt.imshow(w[1:, i].T.reshape((28, -1)), cmap='gray')
        
plot_weights(w)

