
from data_loader import *
import torch
from data_loader import data_generator_np
from data_preparation import data_preparation
from models.MobileNet import MobileNetV3_Small
from models.ResNet32 import ResNet, BasicBlock
import torch.nn as nn
import torch.optim as optim

EPOCH = 100
BATCH_SIZE = 64
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_preparation()
    train_loader, test_loader = data_generator_np(X_train, y_train, X_test, y_test, BATCH_SIZE)

    net = MobileNetV3_Small().to(device)
    # net = ResNet(BasicBlock, [3, 4, 6, 3], 19, 3).to(device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(net):,} trainable parameters')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Test Accuracy：%.3f%%' % (100 * correct / total))






