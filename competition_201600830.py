######################################################################
### For a 2-nd year undergraduate student competition on
### the robustness of deep neural networks, where a student
### needs to develop
### 1. an attack algorithm, and
### 2. an adversarial training algorithm
###
### The score is based on both algorithms.
######################################################################
import copy
import os

import numpy as np
import pandas as pd
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

# adv_train 2: use lib mth to test robustness
import advertorch
from art.defences.trainer import adversarial_trainer

# hyper para tune
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.shcedulers import ASHAScheduler
# from functools import partial

# input id
id_ = 201600830
epsilon = 0.3
alpha = 0.00784

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate') # cascade version
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
# FGSM: num-steps:1 step-size:0.031   PGD-20: num-steps:20 step-size:0.003
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=1,
                    help='perturb number of steps, FGSM: 1, PGD-20: 20')
parser.add_argument('--step-size', default=0.031,
                    help='perturb step size, FGSM: 0.031, PGD-20: 0.003')

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
train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True,
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

## 2021/10/19 version: FGSM_Attack
def fgsm_attack(X, epsilon, data_grad):
    sign_grad = data_grad.sign()
    X_adv = X
    X_adv += epsilon * sign_grad
    X_adv = torch.clamp(X_adv, 0, 1)
    return X_adv


## 2021/10/9 version: LinfPGD attack
class LinPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):  # **
        x = x_natural.detach()
        x += torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(7):
            x.requires_grad = True
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x


def linPGDAttack(x, y, model, adversary):
    adv = adversary.perturb(x, y)
    return adv


def pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
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
    # err_pgd = (model(X_pgd).data.max(1)[1] != y.data).sum()

    # return err, err_pgd
    return X_pgd


# C&W attack: sub-optimal for single un-composite attack
def calc_cw(model, X, y, epsilon=args.epsilon):
    noise = torch.FloatTensor(*X.shape).uniform_(-0.1, 0.1).to(device)
    c = 1e+01
    X_adv = Variable(X.data + noise)
    cw = torch.norm(noise) + c * model(X_adv)
    optimizer = optim.Adam(cw, lr=0.0001)

    X_cw = Variable(X.data, requires_grad=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        X_cw = Variable(X_cw.data + noise, requires_grad=True)
        loss = F.cross_entropy(model(X_cw), target)

        loss.backward()
        optimizer.step()

        eta = torch.clamp(X_cw.data - X.data, -epsilon, epsilon)
        X_cw = Variable(X.data + eta, requires_grad=True)
        X_cw = Variable(torch.clamp(X_cw, 0, 1.0), requires_grad=True)

    return X_cw


# natural attack: Rotation and Translation


'generate adversarial data, you can define your adversarial method'


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data, requires_grad=True)

    #####################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ####################################################################

    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    X_adv = Variable(X_adv.data + random_noise)
    # X_adv = fgsm_attack(X, epsilon, adv_data_grad)
    # X_adv = linPGDAttack(X, y, model, LinPGDAttack(model))
    X_adv = pgd_whitebox(model, X, y)
    return X_adv


#####################################################################
## end of attack method
####################################################################

'train function, you can use adversarial training'
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        adv_data = adv_attack(model, data, target, device=device)

        # clear gradients
        optimizer.zero_grad()

        # loss = F.cross_entropy(model(data), target) # adv train 0 feed adv_x in net training to shape a resilient net
        loss = F.cross_entropy(model(adv_data), target)

        ## ray tune prep
        # output = model(data)
        # pred = output.max(1, keepdim=True)[1]
        # correct += pred.eq(target.view_as(pred)).sum().item()

        # get gradients and update
        loss.backward()
        optimizer.step()

        ## ray tune
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((Net.state_dict(), optimizer.state_dict()), path)
        # tune.report(loss=(loss / len(train_loader.dataset)), accuracy=correct / len(train_loader.dataset))

##############################################################
# advanced adv train 1: cascade adversarial method
# which can produce adversarial images in every mini-batch. Namely, at each batch, it performs a
# separate adversarial training by putting the adversarial images (produced in that batch) into
# the training dataset
##############################################################

##############################################################
# advanced adv train 2: ensemble adversarial training
# which augments training data with perturbations transferred       # from other models.
##############################################################


def cascade_adv_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)


        adv_data = adv_attack(model, data, target, device=device)


        optimizer.zero_grad()

        loss = F.cross_entropy(model(adv_data), target)

        loss.backward()
        optimizer.step()


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
    model = Net().to(device)

    #####################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...).
    ################################################################################################

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)  # adv train 1
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # un-tested!!
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)  # for cascade adv
    # train

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # training
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

        ################################################################################################
        ## end of training method
    ################################################################################################

    # save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model


# hyper-para tune [adv train 0] use during final model train and encap on GPU server
def adjust_learning_rate(optimizer, epoch):
    # lr = args.lr
    # if epoch >= 100:
    #     lr /= 10
    # if epoch >= 150:
    #     lr /= 10

    # cascade version
    lr = 0.1
    if epoch >= 4000:
        lr /= 10
    if epoch >= 6000:
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
        p.append(torch.norm(data - adv_data, float('inf')))
    print('epsilon p: ', max(p))


################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################

# Ray hyper-para tuning works only in linux   TRY ON SERVER
# num_samples=10
# max_num_epochs=10
# config = {
#     "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#     "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([2, 4, 8, 16])
# }
# scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
# reporter = CLIReporter(
# # parameter_columns=["l1", "l2", "lr", "batch_size"],
# metric_columns=["loss", "accuracy", "training_iteration"])
# result = tune.run(
#         partial(train, train_loader),
#         resources_per_trial={"cpu": 2},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter)
#
# best_trial = result.get_best_trial("loss", "min", "last")
# print("Best trial config: {}".format(best_trial.config))
# print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
# print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
# best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])

'Comment out the following command when you do not want to re-train the model'
'In that case, it will load a pre-trained model you saved in train_model()'
model = train_model()

'Call adv_attack() method on a pre-trained model'
'the robustness of the model is evaluated against the infinite-norm distance measure'
'!!! important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!'
p_distance(model, train_loader, device)
