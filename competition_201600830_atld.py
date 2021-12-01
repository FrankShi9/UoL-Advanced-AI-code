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

## adv_train 2: use lib mth to test robustness
# import advertorch
# from art.defences.trainer import adversarial_trainer

## Ray-tune hyper-para tuning
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.shcedulers import ASHAScheduler
# from functools import partial


# ATLD Train
from attack_methods_new_cifar10 import *
from tqdm import tqdm
from WideResnet import *
from dis_atld import *

# input id
id_ = 201600830

epsilon = 0.09
alpha = 0.00784

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
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
# FGSM: num-steps:1 step-size:0.031   PGD-20: num-steps:20 step-size:0.003
parser.add_argument('--epsilon', default=0.031, # scale up
                    help='perturbation')
parser.add_argument('--num-steps', default=1,
                    help='perturb number of steps, FGSM: 1, PGD-20: 20')
parser.add_argument('--step-size', default=0.031,
                    help='perturb step size, FGSM: 0.031, PGD-20: 0.003') # step size < 1/10 eps

args = parser.parse_args(args=[])

# cuda available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda")

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
def fgsm_attack(X, epsilon):
    X = Variable(X.data, requires_grad=True)
    sign_grad = X.sign()
    X_adv = X
    X_adv = Variable(X_adv + (epsilon * sign_grad), requires_grad=True)
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
    # err_pgd = (model(X_pgd).data.max(1)[1] != y.data).sum()       # return err, err_pgd
    return X_pgd


# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(model, X, y, targeted=False, c=1e-4, kappa=0, max_iter=20, learning_rate=0.01, epsilon=args.epsilon):
    images = X.to(device)
    labels = y.to(device)

    # Define f-function
    def f(x):

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1 / 2 * (nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2
        cost = Variable(cost.data, requires_grad=True)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
    attack_images = Variable(torch.clamp(attack_images, -epsilon, epsilon), requires_grad=True)

    return attack_images


# C&W attack: sub-optimal for single un-composite attack
def calc_cw(model, X, epsilon=args.epsilon):
    noise = torch.FloatTensor(*X.shape).uniform_(-0.1, 0.1).to(device)
    c = 1e+01
    X_adv = Variable(X.data + noise)
    X_cw = X_adv
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # change here use marginal loss instead
        X_h = model(X_adv)
        loss = torch.norm(noise, float('inf')) + c * F.cross_entropy(X_h, target)
        optimizer = optim.Adam([noise], lr=0.0001)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)
        X_cw = Variable(X.data + eta, requires_grad=True)
        X_cw = Variable(torch.clamp(X_cw, 0, 1.0), requires_grad=True)

    return X_cw


# GT attack: sub-optimal better than c&w for single un-composite attack
def calc_gt(model, X, X_ac, epsilon=args.epsilon):
    eps_min = 0
    eps_max = epsilon
    t = 1e-4
    x_b = X_ac
    eps = 0.
    x_h = None
    while eps_max - eps_min > t:
        eps = (eps_max + eps_min) / 2
        # !!Invoke Reluplex to test whether ∃x!!
        if x_h is not None:
            eps_max = torch.norm(x_h - X, float('inf'))
            x_b = x_h
        else:
            eps_min = eps

    return x_b


# natural attack: Rotation and Translation # No for now


'generate adversarial data, you can define your adversarial method'


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    #####################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ####################################################################
    # CW
    noise = cw_l2_attack(model, X, y)
    X_adv = Variable(X_adv.data + noise)

    # random noise
    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)

    # X_adv = fgsm_attack(X, epsilon)
    # X_adv = linPGDAttack(X, y, model, LinPGDAttack(model))
    # X_adv = pgd_whitebox(model, X, y)
    # X_adv = calc_cw(model, X)

    # goes with the upper 4
    X_adv = Variable(X_adv.data)

    #####################################################################
    ## end of attack method
    ####################################################################
    return X_adv


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

        loss = F.cross_entropy(model(data), target)
        loss = F.cross_entropy(model(adv_data),
                               target)  # adv train 0 feed adv_x in net training to shape a resilient net

        ##############################################################################################################
        # # ray tune prep
        # output = model(data)
        # pred = output.max(1, keepdim=True)[1]
        # correct += pred.eq(target.view_as(pred)).sum().item()
        ##############################################################################################################

        # get gradients and update
        loss.backward()
        optimizer.step()


#######################################################################################################################
# ray tune
# with tune.checkpoint_dir(epoch) as checkpoint_dir:
#     path = os.path.join(checkpoint_dir, "checkpoint")
#     torch.save((Net.state_dict(), optimizer.state_dict()), path)
# tune.report(loss=(loss / len(train_loader.dataset)), accuracy=correct / len(train_loader.dataset))
#######################################################################################################################

#######################################################################################################################
# advanced adv train 1: cascade adversarial method
# which can produce adversarial images in every mini-batch. Namely, at each batch, it performs a
# separate adversarial training by putting the adversarial images (produced in that batch) into
# the training dataset
#######################################################################################################################

#######################################################################################################################
# advanced adv train 2: ensemble adversarial training
# which augments training data with perturbations transferred       # from other models.
#######################################################################################################################


def cascade_adv_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # correct = 0
    ite = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        adjust_learning_rate_c(optimizer, ite)
        adv_data = adv_attack(model, data, target, device=device)

        optimizer.zero_grad()

        loss = F.cross_entropy(model(adv_data), target)

        loss.backward()

        optimizer.step()

        ite += args.batch_size

        if ite > 8000:
            # early stopping
            return


# ATLD train by Huang
def atld_train(epoch, net):
    # config for feature scatter
    config_feature_scatter = {
        'train': True,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True,
        'ls_factor': 0.5,
    }
    basic_net = WideResNet(depth=28,
                           num_classes=10,
                           widen_factor=10)
    basic_net = basic_net.to(device)
    discriminator = Discriminator_2(depth=28, num_classes=1, widen_factor=5).to(device)
    D_optimizer = optim.SGD(discriminator.parameters(),
                            lr=1e-3,
                            momentum=0.9,
                            weight_decay=0.0001)

    net_org = Attack_FeaScatter(basic_net, config_feature_scatter, discriminator, D_optimizer)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0001)
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < 100:
        lr = args.lr
    elif epoch < 150:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.1 * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        targets = targets.long()
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    adversarial_criterion = nn.BCELoss()
    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # forward
        outputs, loss_fs, gan_loss, scale = net_org(inputs.detach(), targets)

        optimizer.zero_grad()
        loss = loss_fs.mean()
        print('loss_fs:', loss_fs.item())
        # print('gan_loss:', gan_loss.item())
        loss = (loss + gan_loss * scale / 2)
        loss.backward(retain_graph=True)
        for name, parms in net.named_parameters():
            if name == 'module.final_layer.weight':
                max = torch.max(parms.grad)
                min = torch.min(parms.grad)
                diff = (max - min) * 0.3

                max_threshold = max - diff
                min_threshold = min + diff

                parms.grad = parms.grad.clamp(min_threshold, max_threshold)
        optimizer.step()

        train_loss = loss.item()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)
            iterator.set_description(str(adv_acc))

            nat_outputs, _ = net_org(inputs, targets, attack=False)
            nat_acc = get_acc(nat_outputs, targets)


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
    # extra robustness externel toolbox test
    # import foolbox as fb

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

            # model = ...
            # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            # attack = fb.attacks.LinfinityBrendelBethgeAttack()
            # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1]
            # _, advs, success = attack(fmodel, data, target, epsilons=epsilons)
            # print('robustness: ', success)

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


'main function, train the dataset and print train loss, test loss for each epoch'


def train_model():
    model = Net().to(device)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...).
    ################################################################################################

    #  atld only
    config_feature_scatter = {
        'train': True,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True,
        'ls_factor': 0.5,
    }
    basic_net = WideResNet(depth=28,
                           num_classes=10,
                           widen_factor=10)
    basic_net = basic_net.to(device)
    discriminator = Discriminator_2(depth=28, num_classes=1, widen_factor=5).to(device)
    D_optimizer = optim.SGD(discriminator.parameters(),
                            lr=1e-3,
                            momentum=0.9,
                            weight_decay=0.0001)

    net_org = Attack_FeaScatter(basic_net, config_feature_scatter, discriminator, D_optimizer)
    # net_org = torch.nn.DataParallel(net_org)
    net = net_org.basic_net
    discriminator = net_org.discriminator
    for epoch in range(1, 3):
        atld_train(epoch, net)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)  # adv train 0.1

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)  # bad on fgsm feed adv train/ only for c&w solve

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)  # for cascade adv

    # # pre-train 2 epoch given cascade paper for MNIST
    # for epoch in range(1, 3):
    #     start_time = time.time()
    #     # training
    #     adjust_learning_rate(optimizer, epoch)
    #     train(args, model, device, train_loader, optimizer, epoch)

    #################################################################################################################
    # cascade_adv_train(args, model, device, train_loader, optimizer, epoch)
    #################################################################################################################

    # # get trnloss and testloss
    # trnloss, trnacc = eval_test(model, device, train_loader)
    # advloss, advacc = eval_adv_test(model, device, train_loader)
    #
    # # cascade adv
    # print('Pre-train Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
    # print('Pre-train trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
    # print('Pre-train adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

    # save & read the model
    # torch.save(model.state_dict(), str(id_) + '.pt')
    # model_name = str(id_) + '.pt'
    # model = Net()
    # model.load_state_dict(torch.load(model_name))

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # training
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)

        # cascade_adv_train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

        ################################################################################################
        ## end of training method
    #############################################################################################

    ## save the model
    # torch.save(model.state_dict(), str(id_) + '.pt')

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


# Ray tune last stage
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
