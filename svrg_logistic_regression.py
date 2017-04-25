from __future__ import print_function
import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from optim import Eve
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output-path', help='The path where to save the results.',
                    type=str, default='results/cnn/')
parser.add_argument('--update-frequency', help='The frequency to which update the copy of parameters.',
                    type=int, default=5)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = datasets.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

sampler = torch.utils.data.sampler.WeightedRandomSampler(np.ones(len(dataset))/len(dataset), len(dataset))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(dataset,
    batch_size=args.batch_size, sampler=sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class LogisticRegression(nn.Module):
    def __init__(self, l2reg=1e-2):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.l2reg = l2reg
        self.decay_params = list(self.parameters())[0]

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return F.log_softmax(x)

    def get_loss(self, output, target):
        return F.nll_loss(output, target) + self.l2reg * torch.norm(self.decay_params)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

model = CNN()
copy_model = CNN()
copy_model.load_state_dict(model.state_dict())
if args.cuda:
    model.cuda()
    copy_model.cuda()

class SVRG(object):
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.state = defaultdict(dict)

    def reset(self):
        for copy_p in copy_model.parameters():
            state = self.state[copy_p]
            state['mu'] = copy_p.grad.data

    def step(self, data, target):
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        copy_model.zero_grad()
        copy_output = copy_model(data)
        copy_loss = F.nll_loss(copy_output, target)
        copy_loss.backward()

        for p, copy_p in zip(model.parameters(), copy_model.parameters()):
            state = self.state[copy_p]
            print(copy_p.grad.data - state['mu'])
            p.data.add_(-args.lr, p.grad.data - copy_p.grad.data + state['mu'])

        return loss

optimizer = SVRG(args.lr)

def train():
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        loss = optimizer.step(data, target)
        if type(loss) is not float:
            loss = loss.data[0]
        train_loss += loss
    return train_loss/len(train_loader)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader, total=len(test_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    return test_loss/len(test_loader)

filename = os.path.join(args.output_path, 'svrg_'+str(args.lr)+'.csv')

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

s = 'y'
if os.path.isfile(filename):
    s = raw_input("File already exists, do you wanna replace it ? [y,N]")

if s == 'y':
    f = open(filename, 'w')
else:
    exit()

f.write("epoch,train_loss,test_loss\n")
f.flush()

for epoch in range(1, args.epochs + 1):
    if (epoch-1) % args.update_frequency == 0:
        copy_model.load_state_dict(model.state_dict())
        copy_model.zero_grad()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            copy_output = copy_model(data)
            copy_loss = F.nll_loss(copy_output, target)
            copy_loss.backward()
        optimizer.reset()

    train_loss = train()
    test_loss = test()
    print('epoch: %d, train loss: %.4f, test loss: %.4f'%(epoch, train_loss, test_loss))
    f.write("{},{},{}\n".format(epoch, train_loss, test_loss))
    f.flush()
f.close()
