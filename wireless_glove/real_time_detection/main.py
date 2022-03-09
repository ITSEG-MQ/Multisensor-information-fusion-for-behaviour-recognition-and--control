import os
import numpy as np
from data_loader import *
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from data_loader import data_generator_np
from net_main import MainNet

EPOCH = 100
BATCH_SIZE = 64
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    root_path = "./data_fitness"
    files = os.listdir(root_path)
    for fi in files:
        label = np.array(float(fi))
        filepath = os.path.join(root_path, fi)
        files = os.listdir(filepath)
        for file in files:
            sample = np.loadtxt(filepath + "/" + file, delimiter=",")
            for i in range(np.shape(sample)[1]):
                if i * overlapping_rate * window_size + window_size < np.shape(sample)[1]:
                    x = int(i * overlapping_rate * window_size)
                    y = int(i * overlapping_rate * window_size + window_size)
                    sample_.append(sample[0:8, x:y])
                    label_.append(label)

    x = np.array(sample_)
    y = np.array(label_)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    train_loader, test_loader = data_generator_np(X_train, y_train, X_test, y_test, BATCH_SIZE)

    net = MainNet(8, 250, 512, 3).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(net):,} trainable parameters')

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)  # 获取训练数据总长度
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()  # 损失加和（越来越小）
            _, predicted = torch.max(outputs.data, 1)  # 输出这一批次128的对应分类
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()  # 判断这一批次的正确个数，并进行加和
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        with torch.no_grad():  # 里边的数据不需要计算梯度，不需要进行反向传播
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()  # 测试模型时使用该语句，因为模型已经训练完毕，参数不会再更改，所以直接计算训练时所有batch的均值和方差
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类 (outputs.data的索引号)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))         
    torch.save(net, 'net.pt')
