import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                        help='how many batches to wait before logging training status')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False
args.epochs = 3
args.log_interval=50

#%%
kwargs = {}
transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

MAX_DATA = 6000

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
    batch_size=args.batch_size, shuffle=False, **kwargs)

train_loader_stochastic = torch.utils.data.DataLoader(
    train_data,
    batch_size=1, shuffle=False, **kwargs)



test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)

train_data_tensor, train_labels_tensor = iter(train_loader_all).next()
train_data_numpy = train_data_tensor.numpy()
train_labels_numpy = train_labels_tensor.numpy()

#%%
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def get_loss(self, output, target):
        return F.nll_loss(output, target)
    
    def get_example_losses(self, output, target):
        losses = [F.nll_loss(o, t) for o, t in zip(output, target)]
        return losses


class LinearNet(nn.Module):
    def __init__(self, l2reg):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.l2reg = l2reg
        self.decay_params = list(self.parameters())[0]

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return F.log_softmax(x)
    
    def get_loss(self, output, target):
        return F.nll_loss(output, target) + self.l2reg * torch.norm(self.decay_params)

#%%
import utils
reload(utils)
from utils import TrainObserver, TestObserver, plot_smooth
            

def train(epoch, observer=None):
    if observer is None:
        observer = TrainObserver()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = model.get_loss(output, target)
        loss.backward()
        optimizer.step()
        
        observer.update(epoch,
                        batch_idx,
                        len(train_loader),
                        train_loader.batch_size, 
                        loss.data.numpy()[0])
        
        if batch_idx % args.log_interval == 0:
            observer.print_report()


def test(epoch, observer=None): 
    if observer is None:
        observer = TestObserver()
    observer.reset()  # recompute accuracy at each epoch
        
    model.eval()  # because of that, test loss may be smaller than train loss
   
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        output = model(data)
        loss = model.get_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        accuracy = pred.eq(target.data).cpu().sum() / float(test_loader.batch_size)
        
        observer.update(epoch,
                        batch_idx,
                        len(test_loader),
                        test_loader.batch_size, 
                        loss,
                        accuracy)
        
        if batch_idx % args.log_interval == 0:
            observer.print_report()
    observer.print_report()
    
    
def get_losses(epoch, observer=None):
    if observer is None:
        observer = TrainObserver()

    losses = []
    model.train()
    for start_idx in xrange(0, len(train_data_numpy), args.batch_size):
        indices = np.arange(start_idx, start_idx+args.batch_size)
        data = torch.Tensor(train_data_numpy[indices].astype(float))
        target = torch.LongTensor(train_labels_numpy[indices])
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = model.get_loss(output, target)

        batch_losses = model.get_example_losses(output, target)
        batch_losses = [b.data[0] for b in batch_losses]
        losses += list(batch_losses)

        batch_idx = start_idx / args.batch_size
        observer.update(epoch,
                        batch_idx,
                        len(train_loader),
                        train_loader.batch_size, 
                        loss.data.numpy()[0])
        
        if batch_idx % args.log_interval == 0:
            observer.print_report()
            
    return losses


class TrainerNus(object):
    def __init__(self, data, labels, observer=None, exploration=1e-5):
        '''
        Loader should be with batch_size 1
        
        exploration: probability of uniform distribution
        '''
        if observer is None:
            observer = TrainObserver()
            
        self.priorities = np.ones(len(data))
        self.observer = observer
        self.data = data
        self.labels = labels
        self.variances = {}
        self.exploration = exploration
        
    def train(self):     
        model.train()
        for batch_idx in xrange(len(self.data)):
            loss = self.train_batch()
            self.observer.update(epoch,
                                    batch_idx,
                                    len(self.data),
                                    1, 
                                    loss)
                    
            if batch_idx % 600 == 0:
                self.observer.print_report()
                
    def train_batch(self):
        priorities = self.actual_priorities = (self.exploration * np.ones_like(self.priorities) / len(self.priorities)
            + (1-self.exploration) * self.priorities / self.priorities.sum())
        indices = np.random.choice(len(train_data_numpy), p=priorities, size=1)
        data = torch.Tensor(self.data[indices].astype(float))
        target = torch.LongTensor(self.labels[indices])

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = model.get_loss(output, target)
        loss.backward()
        
        example_info = {
                'grad_scale': 1. / (priorities[indices[0]] * len(priorities)),
                'loss': loss.data.numpy()[0]}
        optimizer.step(example_info)
        
        self.update_priorities(indices[0], example_info)
        
        self.variances[indices[0]] = example_info['variation']
        return loss.data.numpy()[0]
    
    def update_priorities(self, index, example_info):
        pass
    
    
class LossTrainer(TrainerNus):
    def __init__(self, *args, **kwargs):
        TrainerNus.__init__(self, *args, **kwargs)
        self.visited = np.zeros_like(self.priorities)
        self.scale = 10.
        self.baseline = 1.
        self.biased_priorities = np.ones_like(self.priorities) * self.baseline
        self.inertia = 0.5
        
    def update_priorities(self, index, example_info):
        #print index, example_info
        new_priority = self.baseline + self.scale * example_info['loss']
        self.biased_priorities[index] = (self.inertia*self.biased_priorities[index] 
            + (1-self.inertia)*new_priority)
        self.priorities[index] = self.biased_priorities[index] # no debiasing for now
        self.visited[index] += 1.
        
    
def get_entropy(p):
    pnz = p[p>0.]
    return -pnz.dot(np.log(pnz))
#%%
model = CNNNet()
#model = LinearNet()

#%% Normal Train
optimizer = optim.Adam(model.parameters())    
train_observer = TrainObserver()
test_observer = TestObserver()
for epoch in xrange(1, args.epochs + 1):
    train(epoch, train_observer)
    test(epoch, test_observer)
    
#%% Normal train using NUS-Train code
import nus_adam
reload(nus_adam)
from nus_adam import NusAdam

optimizer = NusAdam(model.parameters())
train_observer = TrainObserver()
test_observer = TestObserver()
trainer_nus = TrainerNus(train_data_numpy, train_labels_numpy, train_observer)
for epoch in xrange(1, args.epochs + 1):
    trainer_nus.train()
    test(epoch, test_observer)

    
#%% NUS Train
import nus_adam
reload(nus_adam)
from nus_adam import NusAdam

optimizer = NusAdam(model.parameters())
train_observer = TrainObserver()
test_observer = TestObserver()
#trainer_nus = TrainerNus(train_data_numpy, train_labels_numpy, train_observer)
trainer_nus = LossTrainer(train_data_numpy, 
                          train_labels_numpy, 
                          observer=train_observer,
                          exploration=0.1)
for epoch in xrange(1, args.epochs + 1):
    trainer_nus.train()
    test(epoch, test_observer)

        
#%%
optimizer = NusAdam(model.parameters())
optimizer = NusAdam(model.parameters())
train_observer = TrainObserver()
test_observer = TestObserver()
trainer_nus = TrainerNus(train_data_numpy, train_labels_numpy, train_observer)
for epoch in xrange(1, args.epochs + 1):
    trainer_nus.train()
    test(epoch, test_observer)
    
    
#%% Are losses correlated with sample difficulty?
every = 1000
sorted_losses, sorted_data = zip(*sorted(zip(losses[::every], train_data_numpy[::every])))
sorted_data = np.asarray(sorted_data)
torchvision.utils.save_image(torch.Tensor(sorted_data), 'per_loss.png')

#%%
plt.figure(1)
smooth_window = 40
plot_smooth(train_observer.losses, 'Train losses', N=smooth_window)
plot_smooth(nus_train_observer.losses, 'NUS Train losses', N=smooth_window)
plt.legend()

#%%
model_lin_sgd = LinearNet(l2reg=0.001)
model_lin_adam = LinearNet(l2reg=0.001)
model_cnn_sgd = CNNNet()
model_cnn_adam = CNNNet()

#%%
#args.optimizer = 'adam'
args.optimizer = 'sgd'
args.architecture = 'cnn'
#args.architecture = 'linear'

model = None
if args.optimizer == 'sgd':
    if args.architecture == 'linear':
        model = model_lin_sgd
    elif args.architecture == 'cnn':
        model = model_cnn_sgd
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    if args.architecture == 'linear':
        model = model_lin_adam
    elif args.architecture == 'cnn':
        model = model_cnn_adam  
    optimizer = optim.Adam(model.parameters())    

    
print 'Selected', args.optimizer, args.architecture


#%%
for epoch in range(1, args.epochs + 1):
    train_nus(epoch)
    
    
#%%
plt.figure(1)
p = list(model.fc1.parameters())
templates = p[0].data.numpy()
%matplotlib qt
plt.imshow(templates.reshape((10*28, -1)), cmap='gray')
plt.show()
#%%
%matplotlib qt
plt.figure(2)
plt.semilogy(np.arange(len(model.train_losses)), model.train_losses)
plt.show()

#%%
%matplotlib qt

def plot_smooth(losses, label, N=2000):
    smooth_losses = np.convolve(losses, np.ones((N,))/N, mode='valid')
    plt.semilogy(np.arange(len(smooth_losses)), smooth_losses, label=label)


n = max(len(model.train_losses), len(model2.train_losses))
plt.figure(3)
plot_smooth(model_cnn_adam.train_losses, 'model_cnn_adam')
plot_smooth(model_cnn_sgd.train_losses, 'model_cnn_sgd')

#plot_smooth(model2.train_losses, 'model2')
plt.legend()
plt.show()

#%%
plt.figure(4)
plot_smooth(model_cnn_adam.test_acc, 'model_cnn_adam')
plt.show()

#%%
def get_param_size(model):
    sizes = []
    for p in model.parameters():
        sizes.append(np.prod(p.size()))
    return sizes

sizes = get_param_size(model)
print sizes, '-->', np.sum(sizes)