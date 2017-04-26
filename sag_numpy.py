import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_smooth

#%%
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

#%%
X = train_data_numpy.reshape((-1, 28*28))
X = np.hstack((np.ones((len(X), 1)), X))
Y = train_labels_numpy
Y_dummy = np.zeros((len(Y), 10))
Y_dummy[np.arange(len(Y)), Y] = 1.

#%%

def softmax(u):
    '''
    Use exp(u_i) / sum(exp(u_j)) = exp(u_i - u_max) / sum(exp(u_j - u_max))
    '''
    u_max = np.max(u)
    v = u-u_max
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)

def cross_entropy(true, predicted):
    predicted = np.clip(predicted, 1e-15, 1e15)
    return -true.dot(np.log(predicted))


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
    
    
class MultiLogistic(MultiLogisticWeightDecay):
    def __init__(self):
        MultiLogisticWeightDecay.__init__(self, 0.)
    
#%% Parameters
model = MultiLogistic()
nepochs = 2
weight_decay = 0.1

# question: where to put weight decay?

#%% SGD
def sgd(X, Y_dummy, step_size, nepochs, weight_decay):
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    np.random.seed(0)

    for t in xrange(nepochs):
        for i in xrange(len(X)):
    
            i_t = np.random.randint(len(X))
            x = X[i_t]
            y = Y_dummy[i_t]
            # compute gradient and loss
            grad_scalar, grad, loss = model.grad_and_loss(w, x, y)
            # update
            w = w - step_size * (grad + float(weight_decay)/w.size*w)
            total_loss = loss + 0.5*weight_decay*np.mean(w*w)
            # update
            losses.append(total_loss)
    
            if (i+1) % 10000 == 0:
                print 'Epoch {}. {}/{} -- train loss {}'.format(t, i+1, len(X), np.mean(losses))
                
    return w, losses

step_size = 0.1
w_sgd, losses_sgd = sgd(X, Y_dummy, step_size, nepochs, weight_decay)
w = w_sgd
losses = losses_sgd

#%% SAGA
def saga(X, Y_dummy, step_size, nepochs, weight_decay):
    losses = []
    w = np.zeros((X.shape[1], Y_dummy.shape[1]))
    np.random.seed(0)

    memories = np.zeros((X.shape[0], Y_dummy.shape[1]))
    sum_memories = np.zeros_like(w)
    visited = np.zeros(X.shape[0])

    for t in xrange(nepochs):
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
            memories[i_t, :] = grad_scalar
            sum_memories = sum_memories + grad - old_grad
            w = w - step_size * (sag_update + float(weight_decay)/w.size*w)
            total_loss = loss + 0.5*weight_decay*np.mean(w*w)
            
            # accumulate
            losses.append(total_loss)
    
            if (i+1) % 10000 == 0:
                print 'Epoch {}. {}/{} -- train loss {}'.format(t, i+1, len(X), np.mean(losses))
                
    return w, losses

step_size = 0.1
w_saga, losses_saga = saga(X, Y_dummy, step_size, nepochs, weight_decay)
w = w_saga
losses = losses_saga

#%% prediction
Y_pred = np.argmax(X.dot(w), axis=1)
accuracy = np.mean(Y_pred == Y)
print 'accuracy', accuracy

#print losses
#%% Plot smooth
plot_smooth(losses, 'train loss', N=1000)

#%%
plot_smooth(losses_sgd, 'SGD', N=1000)
plot_smooth(losses_saga, 'SAGA', N=1000)
plt.legend()

#%% did it learn?
def plot_weights(w):
    for i in xrange(10):
        plt.subplot(4,3,i+1)
        plt.imshow(w[1:, i].T.reshape((28, -1)), cmap='gray')
        
plot_weights(w)