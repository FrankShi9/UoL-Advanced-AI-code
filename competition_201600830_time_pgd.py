######################################################################
### For a 2-nd year undergraduate student competition on
### the robustness of deep neural networks, where a student
### needs to develop
### 1. an attack algorithm, and
### 2. an adversarial training algorithm
###
### The score is based on both algorithms.
######################################################################
import os
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

# Ray-tune hyper-para tuning
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.shcedulers import ASHAScheduler
# from functools import partial



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
parser.add_argument('--num-steps', default=40,
                    help='perturb number of steps, FGSM: 1, PGD-20: 20') # change from 1 to 20 3/12/2021 -> to 50 -> 100 -> 40
parser.add_argument('--step-size', default=0.011, # change from 0.031 to 0.1099 1/12/2021 -> from 0.1099 to 0.005495 -> from 0.005495 to 0.011 on 3/12/2021
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
train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
# test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True,
#                                              transform=transforms.Compose([transforms.ToTensor()]))
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
# test_set = torchvision.datasets.EMNIST(root='../data', split='letters', train=False, download=True,
#                                              transform=transforms.Compose([transforms.ToTensor()]))
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
# test_set = torchvision.datasets.KMNIST(root='../data', train=False, download=True,
#                                              transform=transforms.Compose([transforms.ToTensor()]))
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

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
    start_time = time.time()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    i = 0
    # timecost_now = int(time.time() - start_time) uses 3s
    timecost_now = 0

    while i < num_steps and timecost_now < 0.1:
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
        time_now = time.time()
        timecost_now = float(time_now - start_time)
        i += 1

    print(timecost_now, i)
    return X_pgd



'generate adversarial data, you can define your adversarial method'


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    #####################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ####################################################################

    X_adv = pgd_whitebox(model, X, y)

    #wrap up
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

        #######################################################################################################################
        ## ray tune prep
        # output = model(data)
        # pred = output.max(1, keepdim=True)[1]
        # correct += pred.eq(target.view_as(pred)).sum().item()
        #######################################################################################################################

        # get gradients and update
        loss.backward()
        optimizer.step()


# ray tune
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
    #debug only
    cnt = 0

    with torch.no_grad():
        for data, target in test_loader:
            cnt += 1
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            print('batch: ', cnt)
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

    ## toolbox test
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)  # adv train 0.1

    for epoch in range(1, args.epochs + 1):
        start_time = time.time() # time is accurate

        ## training only
        # adjust_learning_rate(optimizer, epoch) # adv_train 1.0
        # train(args, model, device, train_loader, optimizer, epoch)

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

