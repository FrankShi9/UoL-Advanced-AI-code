import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import time

print(torch.__version__)

#setup hyper param
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',  help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',  help='input batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',  help='number of epochs to train')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  help='learning rate')

parser.add_argument('--no-cuda', action='store_true', default=False,  help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',  help='random seed (default: 1)')

args = parser.parse_args(args=[])

device = torch.device("cpu")

torch.manual_seed(args.seed)

#setup dataloader
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
testset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)



#define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

#train func
def train(args, model, device, train_loader, optimizer, epoch):
    model.train() #super class method
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)

        optimizer.zero_grad()

        # forward
        loss = F.cross_entropy(model(data), target)

        # backward
        loss.backward()
        optimizer.step()


#predict func
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28*28)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def main():
    model = Net().to(device)
    print(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train(args, model, device, train_loader, optimizer, epoch)

        train_loss, train_acc = eval_test(model, device, train_loader)
        test_loss, test_acc = eval_test(model, device, test_loader)

        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(train_loss, 100*train_acc), end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(test_loss, 100*test_acc))
        # print('train loss', train_loss)
        # print('test loss', test_loss)

if __name__ == '__main__':
    main()

