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
from art.estimators.classification import PyTorchClassifier
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time

# adv_train 2: use lib mth to test robustness
# import advertorch
# from art.defences.trainer import adversarial_trainer

# Ray-tune hyper-para tuning
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.shcedulers import ASHAScheduler
# from functools import partial


# ATLD Train
from attack_methods_new_cifar10 import *
from tqdm import tqdm
from WideResnet import *
from dis_atld import *

#auto attack test
from art.attacks.evasion import AutoAttack


# input id
id_ = 201600830


# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--epochs', type=int, default=1, metavar='N', # 1 for test, 10 for train
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
parser.add_argument('--num-steps', default=100,
                    help='perturb number of steps, FGSM: 1, PGD-20: 20') # change from 1 to 20 3/12/2021 -> to 50 -> 100 -> 40
parser.add_argument('--step-size', default=0.011, # change from 0.031 to 0.1099 1/12/2021 -> from 0.1099 to 0.005495 -> from 0.005495 to 0.011 on 3/12/2021
                    help='perturb step size, FGSM: 0.1099, PGD-20: 0.005495') # change from 0.1099 to 0.005495 3/12/2021
#ODS only
parser.add_argument('--ODI-num-steps', default=2, type=int,
                    help='ODI perturb number of steps')
parser.add_argument('--ODI-step-size', default=8/255, type=float,
                    help='ODI perturb step size')

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



def pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size, ODI_num_steps=args.ODI_num_steps,
                  ODI_step_size=args.step_size):
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    randVector = torch.FloatTensor(*model(X_pgd).shape).uniform_(-1., 1.).to(device)

    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(ODI_num_steps + num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()


        with torch.enable_grad():
            if i < ODI_num_steps:
                loss = (model(X_pgd) * randVector).sum()
            # elif args.lossFunc == 'xent':
            #     loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            else:
                loss = margin_loss(model(X_pgd), y)
        loss.backward()

        if i < ODI_num_steps:
            eta = ODI_step_size * X_pgd.grad.data.sign()
        else:
            eta = step_size * X_pgd.grad.data.sign()

        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd

# ODS
def margin_loss(logits,y):

    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cpu") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss



'generate adversarial data, you can define your adversarial method'


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    # for ART auto only
    # X = Variable(X.float(), requires_grad=True)
    # y = Variable(y.float(), requires_grad=True)

    #####################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ####################################################################

    ## random noise mth
    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)


    # CW
    # noise = cw_l2_attack(model, X, y)
    # X_adv = Variable(X_adv.data + noise)



    ## method combo
    X_adv = pgd_whitebox(model, X, y)
    # X_adv = fgsm_attack(X, args.epsilon)
    # X_adv = linPGDAttack(X, y, model, LinPGDAttack(model))

    ## untested below
    # X_adv = calc_cw(model, X)

    ## AutoAttack by ART ##bug!!##
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    # classifier = PyTorchClassifier(
    #     model=model,
    #     clip_values=(0, 255),
    #     loss=criterion,
    #     optimizer=optimizer,
    #     input_shape=(1, 784),
    #     nb_classes=10,
    # )
    # auto = AutoAttack(estimator=classifier, eps=0.1099, batch_size=128)
    # X_adv = torch.tensor(auto.generate(X.view(X.size(0), 28 * 28).detach().numpy(), y.detach().numpy()))

    # # goes with the upper 4 ones
    X_adv = Variable(X_adv.data)

    # tas auto
    # import torchattacks
    # cw = torchattacks.CW(model)
    # auto = torchattacks.AutoAttack(model, eps=args.epsilon)
    # X_adv = auto(X, y)
    # X_adv = Variable(X_adv.data)
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

        # loss = F.cross_entropy(model(data), target)
        loss = F.cross_entropy(model(adv_data),
                               target)  # adv train 0 feed adv_x in net training to shape a resilient net

        #######################################################################################################################
        ## ray tune prep
        # output = model(data)
        # pred = output.max(1, keepdim=True)[1]
        # correct += pred.eq(target.view_as(pred)).sum().item()
        #######################################################################################################################

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
    model = Net()
    model_name = str(id_) + '.pt'
    model.load_state_dict(torch.load(model_name))
    model.to(device)

    ## normal one (send)
    # model = Net().to(device)

##########################################################################
    # # toolbox test
    # import foolbox as fb
    # fmodel = fb.PyTorchModel(model, bounds=(0, 255))
    #
    # import foolbox.attacks.fast_gradient_method as fgsm
    # import foolbox.attacks.deepfool as df
    # import foolbox.attacks.carlini_wagner as cw
    # import foolbox.attacks.projected_gradient_descent as pgd
    # import foolbox.attacks.brendel_bethge as bb
    # import foolbox.attacks.newtonfool as new
    # import foolbox.attacks.ead as ead
    # import foolbox.attacks.ddn as ddn
    # import foolbox.attacks.gradient_descent_base as gd
    # import foolbox.attacks.virtual_adversarial_attack as va
    # import foolbox.attacks.spatial_attack as sa
    # import foolbox.attacks.saltandpepper as sp
    #
    #
    # attack = fgsm.LinfFastGradientAttack()
    # attack = df.LinfDeepFoolAttack()
    # attack = pgd.LinfProjectedGradientDescentAttack()
    # # attack = cw.L2CarliniWagnerAttack() # cannot run since take too long
    # # attack = ead.EADAttack() # cannot run since take too long
    # # attack = bb.LinfinityBrendelBethgeAttack() # assertion error
    # # attack = new.NewtonFoolAttack()
    # # attack = ddn.DDNAttack()
    # # attack = gd.LinfBaseGradientDescent(rel_stepsize=20, steps=20, random_start=True)
    # # attack = va.VirtualAdversarialAttack(steps=20)
    # # attack = sa.SpatialAttack() #only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)
    # # attack = sp.SaltAndPepperNoiseAttack()
    #
    #
    # # epsilons = [0.001, 0.01, 0.03, 0.1]
    # epsilons = [0.1099]
    #
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     data, target = data.to(device), target.to(device)
    #     data = data.view(data.size(0), 28 * 28)
    #     _, advs, success = attack(fmodel, data, target, epsilons=epsilons)
    #     # print('robustness: ', success)
    #     print('effective attack: ', len([i for i in success[0] if i == True])/len(success[0]))
    #     print(fb.utils.accuracy(fmodel, data, target))
###########################################################################


    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...).
    ################################################################################################

    #  atld only
    # config_feature_scatter = {
    #     'train': True,
    #     'epsilon': 8.0 / 255 * 2,
    #     'num_steps': 1,
    #     'step_size': 8.0 / 255 * 2,
    #     'random_start': True,
    #     'ls_factor': 0.5,
    # }
    # basic_net = WideResNet(depth=28,
    #                        num_classes=10,
    #                        widen_factor=10)
    # basic_net = basic_net.to(device)
    # discriminator = Discriminator_2(depth=28, num_classes=1, widen_factor=5).to(device)
    # D_optimizer = optim.SGD(discriminator.parameters(),
    #                         lr=1e-3,
    #                         momentum=0.9,
    #                         weight_decay=0.0001)
    #
    # net_org = Attack_FeaScatter(basic_net, config_feature_scatter, discriminator, D_optimizer)
    # # net_org = torch.nn.DataParallel(net_org)
    # net = net_org.basic_net
    # discriminator = net_org.discriminator
    # for epoch in range(1, 3):
    #     atld_train(epoch, net)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)  # adv train 0.1

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)  # bad on fgsm feed adv train/ only for c&w solve

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)  # for cascade adv
    #
    # # # pre-train 2 epoch given cascade paper for MNIST
    # for epoch in range(1, 3):
    #     start_time = time.time()
    #     # training
    #     # adjust_learning_rate(optimizer, epoch)
    #     # train(args, model, device, train_loader, optimizer, epoch)
    # #################################################################################################################
    #     cascade_adv_train(args, model, device, train_loader, optimizer, epoch)
    # #################################################################################################################
    #
    #     # get trnloss and testloss
    #     trnloss, trnacc = eval_test(model, device, train_loader)
    #     advloss, advacc = eval_adv_test(model, device, train_loader)
    #
    #     # cascade adv train
    #     print('Pre-train Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
    #     print('Pre-train trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
    #     print('Pre-train adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))
    # #################################################################################################################
    # ## save pre-trained model
    # torch.save(model.state_dict(), str(id_) + '.pt')

    ## read pre-trained model
    # model_name = str(id_) + '.pt'
    # model = Net()
    # model.load_state_dict(torch.load(model_name))
    #################################################################################################################

    for epoch in range(1, args.epochs + 1):
        start_time = time.time() # time is accurate

        ## training only
        # adjust_learning_rate(optimizer, epoch) # adv_train 1.0
        # train(args, model, device, train_loader, optimizer, epoch)
        ##################################################################################
        # cascade_adv_train(args, model, device, train_loader, optimizer, epoch)
        ##################################################################################

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

    ## save the final model
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

