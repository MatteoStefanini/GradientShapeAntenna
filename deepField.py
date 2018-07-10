import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter


class Antennas(Dataset):
    def __init__(self, data, fitness):
        self.data = data
        self.fitness = fitness

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item]).requires_grad_(), torch.from_numpy(self.fitness[item])

    def __len__(self):
        return len(self.data)


class DeepField(nn.Module):
    def __init__(self):
        super(DeepField, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.avg = nn.AvgPool2d(5, stride=2)
        self.fc = nn.Linear(160, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.avg(x)

        x = self.fc(x.view(x.shape[0], -1))

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeepResField(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 16
        super(DeepResField, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)

        self.avg = nn.AvgPool2d(3, stride=2)
        self.fc = nn.Linear(96, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avg(x)
        x = self.fc(x.view(x.size(0), -1))

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.Conv2d(planes * block.expansion, planes * block.expansion, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def read_dataset_onefitness(folder=None):
    dt = np.dtype([('combination', 'uint8', (504,))])
    records = np.fromfile('dataset_pop_260618_1642.dat', dt)
    data = np.concatenate(records.tolist(), axis=0)
    print(data.shape)
    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1) \
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)

    dt = np.dtype([('fitness', 'float32', (1,))])
    fit = np.fromfile('dataset_fitness_260618_1642.dat', dt)
    fitness = np.concatenate(fit.tolist(), axis=0)
    print(fitness.shape)

    return antennas, fitness


def read_dataset(folder=None):
    dt = np.dtype([('combination', 'uint8', (504,))])
    records = np.fromfile('dataset_pop_090718_1705.dat', dt)
    data = np.concatenate(records.tolist(), axis=0)
    print(data.shape)
    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1) \
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)

    dt = np.dtype([('field', 'float32', (171,))])
    field_records = np.fromfile('dataset_FF_090718_1705.dat', dt)
    print(field_records.shape)
    field = np.concatenate(field_records.tolist(), axis=0)
    print(field.shape)

    return antennas, field


def plot_field(field):
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d

    Phi = np.radians(np.linspace(-90, 90, 19))
    Theta = np.radians(np.linspace(1, 90, 9))
    THETA, PHI = np.meshgrid(Theta, Phi)

    R = field.reshape((len(Phi), len(Theta)))

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'),  # or jet
        linewidth=0.1, antialiased=False, alpha=0.5)

    plt.show()


def training(antennas, fitness):
    writer = SummaryWriter()

    data_train, data_val, fit_train, fit_val = train_test_split(antennas, fitness, test_size=0.20, random_state=7)
    dataset_train = Antennas(data_train, fit_train)
    dataloader_train = DataLoader(dataset_train, batch_size=32, num_workers=0)
    dataset_val = Antennas(data_val, fit_val)
    dataloader_val = DataLoader(dataset_val, batch_size=32, num_workers=0)

    #model = DeepField().to(device)
    model = DeepResField(Block, [1, 1]).to(device)
    print(model)
    #writer.add_graph(model, )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

    for epoch in range(50):
        print("Epoch: %d" % (epoch+1))
        loss_train = 0.0; count = 0
        model.train()
        dataloader_iter = iter(dataloader_train)
        for it, (antennas, fitness) in enumerate(dataloader_iter):
            antennas = antennas.float().to(device)
            fitness = fitness.to(device)

            output = model(antennas)
            loss = criterion(output, fitness)
            loss_train += loss.item()
            count += len(antennas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Training epoch {}, loss {}'.format(epoch, loss_train/count))
        writer.add_scalar('Train/Loss', loss_train/count, epoch)

        # EVALUATION
        loss_eval = 0.0; count = 0
        model.eval()
        dataloader_val_iter = iter(dataloader_val)
        for it, (antennas, fitness) in enumerate(dataloader_val_iter):
            antennas = antennas.float().to(device)
            fitness = fitness.to(device)

            output = model(antennas)
            loss = criterion(output, fitness)
            loss_eval += loss.item()
            count += len(antennas)

        print('Evaluation epoch {}, loss {}'.format(epoch, loss_eval/count))
        writer.add_scalar('Val/Loss', loss_eval/count, epoch)
    writer.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # antennas, fitness = read_dataset_onefitness()
    antennas, fields = read_dataset()

    plot_field(fields[0])

    training(antennas, fields)
