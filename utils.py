import numpy as np
import matplotlib.pyplot as plt


class Smoother(object):
    def __init__(self, beta=0.95):
        self.beta = beta
        self.reset()
        
    def reset(self):
        self.smoothed = 0
        self.n_vals = 0
        
    def update(self, val):
        self.smoothed = self.smoothed * self.beta + (1-self.beta) * val
        self.n_vals += 1
        
    def get(self):
        unbiased = self.smoothed / (1 - self.beta ** self.n_vals)
        return unbiased
    
    
class TrainObserver(object):
    def __init__(self, loss_smoother=None):
        if loss_smoother is None:
            loss_smoother = Smoother()
        self.loss_smoother = loss_smoother
        self.reset()
        
    def reset(self):
        self.loss_smoother.reset()
        self.losses = []
        
    def update(self, epoch, batch_idx, n_batches, batch_size, loss):
        self.loss_smoother.update(loss)
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.losses.append(loss)
    
    def get_loss(self):
        return self.loss_smoother.get()
        
    def print_report(self):
        try:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.epoch, 
                self.batch_idx * self.batch_size, 
                self.n_batches*self.batch_size,
                100. * self.batch_idx / self.n_batches, 
                self.get_loss()))
        except Exception as e:
            print e
            print 'TrainObserver --> call update() before print_report()'
            
                                                
class TestObserver(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = []
        self.accuracies = []
        
    def update(self, epoch, batch_idx, n_batches, batch_size, loss, accuracy):
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def get_loss(self):
        return np.mean(self.losses)
    
    def get_accuracy(self):
        return np.mean(self.accuracies)
        
    def print_report(self):
        try:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                self.epoch, 
                self.batch_idx * self.batch_size, 
                self.n_batches*self.batch_size,
                100. * self.batch_idx / self.n_batches, 
                self.get_loss(),
                self.get_accuracy()))
        except Exception as e:
            print e
            print 'TestObserver --> call update() before print_report()'
            
                                               
def plot_smooth(losses, label, N=2000):
    smooth_losses = np.convolve(losses, np.ones((N,))/N, mode='valid')
    plt.semilogy(np.arange(len(smooth_losses)), smooth_losses, label=label)

