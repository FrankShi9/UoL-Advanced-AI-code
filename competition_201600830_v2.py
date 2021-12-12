######################################################################
### For a 2-nd year undergraduate student competition on
### the robustness of deep neural networks, where a student
### needs to develop
### 1. an attack algorithm, and
### 2. an adversarial training algorithm
###
### The score is based on both algorithms.
######################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time

# input id
id_ = 201600830


# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--epochs', type=int, default=30, metavar='N', # 1 for test, 10 for train
                    help='number of epochs to train')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')  # cascade version, original = 0.01
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
# FGSM: num-steps:1 step-size:0.1099   PGD-20: num-steps:20 step-size:0.005495
parser.add_argument('--epsilon', default=0.1099, # change from 0.031 to 0.1099 1/12/2021
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps, FGSM: 1, PGD-20: 20') # change from 1 to 20 3/12/2021 -> to 50 -> 100 -> 40
parser.add_argument('--step-size', default=0.05, # change from 0.031 to 0.1099 1/12/2021 -> 0.005495 -> 0.011 3/12/2021 -> 0.05 on 12/12/2021
                    help='perturb step size, FGSM: 0.1099, PGD-20: 0.005495') # change from 0.1099 to 0.005495 3/12/2021

args = parser.parse_args(args=[])

# cuda available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

######################################################################
################    don't change the below code
######################################################################
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):  # x is a tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


######################################################################
#############    end of "don't change the below code"
######################################################################


def pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    return X_pgd

def pgd_whitebox60(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(60):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    return X_pgd

# natural attack: Rotation and Translation # No for now


'generate adversarial data, you can define your adversarial method'


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

#####################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ####################################################################

    X_adv = pgd_whitebox(model, X, y)   # adv train only
    # X_adv = pgd_whitebox60(model, X, y) # attack only
    # wrap up
    X_adv = Variable(X_adv.data)
#####################################################################
    ## end of attack method
    ####################################################################
    return X_adv


'train function, you can use adversarial training'

def train(args, scheduler, model, device, train_loader, optimizer, epoch):
    model.train()
    lrs = []
    iters = len(train_loader)
    counter = 0
    # correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += 1
        lrs.append(scheduler.get_last_lr()[0])
        # print the LR after each 500 iterations
        if counter % 500 == 0:
            print(f"[INFO]: LR at iteration {counter}: {scheduler.get_last_lr()}")

        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        adv_data = adv_attack(model, data, target, device=device)

        # clear gradients
        optimizer.zero_grad()

        # loss = F.cross_entropy(model(data), target)
        loss = F.cross_entropy(model(adv_data),
                               target)  # adv train 0 feed adv_x in net training to shape a resilient net

        # get gradients and update
        loss.backward()
        optimizer.step()

        scheduler.step(epoch + batch_idx / iters)

'predict function'


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_adv_test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


'main function, train the dataset and print train loss, test loss for each epoch'


def train_model():

    ## test one (read and load)
    # model = Net()
    # model_name = str(id_) + '.pt'
    # model.load_state_dict(torch.load(model_name))
    # model.to(device)

    ## normal one (send)
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                          weight_decay=0.0005)

    # when using warm restarts

    print('[INFO]: Initializing Cosine Annealing with Warm Restart Scheduler')
    steps = 5
    mult = 1
    print(f"[INFO]: Number of epochs for first restart: {steps}")
    print(f"[INFO]: Multiplicative factor: {mult}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps,
            T_mult=mult
        )



    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)  # adv train 0.1
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)  # adv train 0.5

#################################################################################################################

    for epoch in range(1, args.epochs + 1):
        start_time = time.time() # time is accurate

        ## training only
        # adjust_learning_rate(optimizer, epoch) # adv_train 1.0
        train(args, scheduler, model, device, train_loader, optimizer, epoch)
        ##################################################################################
##################################################################################

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print(f"[INFO]: Current LR [Epoch Begin]: {scheduler.get_last_lr()}")
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

    # official ability test 12/11
    adv_tstloss, adv_tstacc = eval_adv_test(model, device, test_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(
        1 / adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(
        adv_tstacc))    ################################################################################################
        ## end of training method
        #############################################################################################

    ## save the final model
    torch.save(model.state_dict(), str(id_) + '.pt')

    return model


# hyper-para tune [adv train 0] use during final model train and encap on GPU server
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_c(optimizer, ite):
    lr = args.lr
    if ite >= 4000:
        lr /= 10
    if ite >= 6000:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'compute perturbation distance'


def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data - adv_data, float('inf')))  # infinity norm
    print('epsilon p: ', max(p))


'Comment out the following command when you do not want to re-train the model'
'In that case, it will load a pre-trained model you saved in train_model()'

model = train_model()

'Call adv_attack() method on a pre-trained model'
'the robustness of the model is evaluated against the infinite-norm distance measure'
'!!! important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!'
p_distance(model, train_loader, device)

