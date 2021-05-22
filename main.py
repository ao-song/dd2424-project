from model import Colorizer
from preprocess import lab_training_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch import split
import torch
from ultils import soft_encoding_ab
import matplotlib.pyplot as plt


def plot(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iteration')
    # plt.ylabel('CrossEntropy Loss')
    # plt.savefig('ce_loss.png')
    plt.ylabel('MSE Loss')
    plt.savefig('mse_loss.png')



Colorizer = Colorizer()

# No gpu..
# Colorizer.cuda()

optimizer = optim.Adam(Colorizer.parameters(), lr=0.001)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

epochs = 80
# batch_size = 100
loss_list = []

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(lab_training_loader):

        # print(data.shape)
        lab = split(data, [1, 2], dim=1)
        l = lab[0]
        ab = lab[1]

        # print(l.shape)
        # print(ab.shape)

        l = Variable(l)
        ab = Variable(ab)

        # target = torch.from_numpy(soft_encoding_ab(ab.detach().numpy()))
        # print(target.shape)

        output = Colorizer(l)
        # output = torch.from_numpy(soft_encoding_ab(output.detach().numpy()))
        # print(output)
        # print(output.shape)

        optimizer.zero_grad()
        loss = criterion(output, ab)
        loss.backward()
        optimizer.step()

        running_loss += (loss.item())
    loss_list.append(running_loss)
    print(f'Running loss is: {running_loss}')

    torch.save(Colorizer.state_dict(), 'models/cifar10_colorizer'+str(epoch))

plot(loss_list)


