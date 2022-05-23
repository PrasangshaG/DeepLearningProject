import imp
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time

log_interval = 180
STEP = 1


def train(model, epoch, train_loader, loss_func, optimizer, device):
    model.train()
    training_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_func(output, target).to(device)
        correct += (torch.max(output, 1)
                    [1].view(target.size()).data == target.data).sum().item()
        training_loss += loss.item()
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    training_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        training_loss, correct, len(train_loader.dataset), accuracy))

    return accuracy, training_loss


def test(model, test_loader, loss_func, device):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data).to(
                device), Variable(target).to(device)
            output = model(data)
            # sum up batch loss
            validation_loss += loss_func(output,
                                         target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    validation_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy, validation_loss


def evaluate(epochs, train_loader, test_loader, model, optimizer, device, microNet=False, scheduler=None):
    train_a, train_l = [], []
    test_a, test_l = [], []
    start_time = time.time()
    for epoch in range(1, epochs+1):
        train_acc, train_loss = train(
            model, epoch, train_loader, F.nll_loss, optimizer, device)
        train_a.append(train_acc)
        train_l.append(train_loss)
        test_acc, test_loss = test(model, test_loader, F.nll_loss, device)
        test_a.append(test_acc)
        test_l.append(test_loss)
        if microNet:
            if epoch % STEP:
                scheduler.step()
    end_time = time.time()
    return (max(train_a), max(test_a), epochs, end_time-start_time), [train_a, train_l, test_a, test_l]
