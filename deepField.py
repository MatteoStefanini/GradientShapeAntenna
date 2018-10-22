import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import os, math, test, time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter


class Antennas(Dataset):
    def __init__(self, data, field):
        self.data = data
        self.field = field

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item]).requires_grad_(), torch.from_numpy(self.field[item])

    def __len__(self):
        return len(self.data)


class DeepField(nn.Module):
    def __init__(self):
        super(DeepField, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.avg = nn.AvgPool2d(5, stride=2)
        self.fc = nn.Linear(320, 153)

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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=16)
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
        self.layer2 = self._make_layer(block, 32, layers[1])
        self.layer3 = self._make_layer(block, 48, layers[2], stride=2)

        #self.avg = nn.AvgPool2d(3, stride=2)
        self.fc = nn.Linear(1536, 153)

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
        x = self.layer3(x)

        #x = self.avg(x)
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


def read_dataset(folder=None):
    dt = np.dtype([('combination', 'uint8', (504,))])
    records1 = np.fromfile('datasetfarfield/dataset_pop_130918_1646.dat', dt)
    records2 = np.fromfile('datasetfarfield/dataset_pop_260918_0054.dat', dt)
    tmp1 = np.concatenate(records1.tolist(), axis=0)
    tmp2 = np.concatenate(records2.tolist(), axis=0)
    data = np.concatenate((tmp1, tmp2), axis=0)
    print(data.shape)

    mapping = np.genfromtxt('mapping_new.csv', delimiter=',', dtype=np.int16)
    mapping = np.subtract(mapping, 1)  # matlab index is evil
    data = data[:, mapping]

    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1) \
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)

    dt = np.dtype([('field', '<f4', (153,))])
    field_records1 = np.fromfile('datasetfarfield/dataset_FF_130918_1646.dat', dt)
    field_records2 = np.fromfile('datasetfarfield/dataset_FF_260918_0054.dat', dt)
    ftmp1 = np.concatenate(field_records1.tolist(), axis=0)
    ftmp2 = np.concatenate(field_records2.tolist(), axis=0)
    fields = np.concatenate((ftmp1, ftmp2), axis=0)
    print(fields.shape)

    return antennas, fields


def read_datasets(folder=None):
    pops = list()
    popfiles = [f for f in os.listdir('populations/') if os.path.isfile(os.path.join('populations/', f))]
    dt = np.dtype([('combination', 'uint8', (504,))])
    for pop in popfiles:
        records = np.fromfile(os.path.join('populations/', pop), dt)
        tmp = np.concatenate(records.tolist(), axis=0)
        pops.append(tmp)
    data = np.concatenate(pops, axis=0)
    print(data.shape)

    mapping = np.genfromtxt('mapping_new.csv', delimiter=',', dtype=np.int16)
    mapping = np.subtract(mapping, 1)  # matlab index is evil
    data = data[:, mapping]

    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1) \
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)

    ffs = list()
    fffiles = [f for f in os.listdir('farfields/') if os.path.isfile(os.path.join('farfields/', f))]
    dt = np.dtype([('field', '<f4', (153,))])
    for ff in fffiles:
        field_records = np.fromfile(os.path.join('farfields/', ff), dt)
        ftmp = np.concatenate(field_records.tolist(), axis=0)
        ffs.append(ftmp)
    fields = np.concatenate(ffs, axis=0)
    print(fields.shape)

    return antennas, fields


def training(antennas, field, save=False):
    writer = SummaryWriter()

    data_train, data_val, fit_train, fit_val = train_test_split(antennas, field, test_size=0.20, random_state=7)
    dataset_train = Antennas(data_train, fit_train)
    dataloader_train = DataLoader(dataset_train, batch_size=128, num_workers=0)
    dataset_val = Antennas(data_val, fit_val)
    dataloader_val = DataLoader(dataset_val, batch_size=128, num_workers=0)

    #model = DeepField().to(device)
    model = DeepResField(Block, [2, 3, 1]).to(device)
    print(model)
    print('learnable parameters: {}'.format(test.count_parameters(model)))
    criterion = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=0.000013, weight_decay=0.38)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
    schedulerStep = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(18):
        schedulerStep.step()
        print("Epoch: %d" % epoch)
        loss_train = 0.0; count = 0
        model.train()
        dataloader_iter = iter(dataloader_train)
        for it, (antennas, field) in enumerate(dataloader_iter):
            antennas = antennas.float().to(device)
            field = field.to(device)

            output = model(antennas)
            loss = criterion(output, field)
            loss_train += loss.item()
            count += 1  # count += len(antennas)

            if epoch == 0 and count < 3:
                print('first_loss_train: ', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Training epoch {}, loss {}'.format(epoch, loss_train/count))
        writer.add_scalar('DeepResTrain/Loss', loss_train/count, epoch)

        # EVALUATION
        loss_eval = 0.0; count = 0; best_loss = 9999
        model.eval()
        dataloader_val_iter = iter(dataloader_val)
        with torch.no_grad():
            for it, (antennas, field) in enumerate(dataloader_val_iter):
                antennas = antennas.float().to(device)
                field = field.to(device)

                output = model(antennas)
                loss = criterion(output, field)
                loss_eval += loss.item()
                count += 1  # count += len(antennas)

            scheduler.step(loss_eval/count)
            print('Evaluation epoch {}, loss {}'.format(epoch, loss_eval/count))
            writer.add_scalar('DeepResVal/Loss', loss_eval/count, epoch)

            if loss_eval/count < best_loss:
                best_loss = loss_eval/count
                if save:
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),
                    }, 'checkpoint.pt.tar')

    writer.close()
    return model


def shapeOptimizer(input_antenna, numsteps=10000, load=False, model=None):
    target = torch.Tensor([60,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).to(device)
    target = target.expand(input_antenna.size(0), 153)
    # input_antenna = random
    test.draw_antenna(input_antenna[0].numpy(), 'input_antenna1.jpg')
    input_antenna = input_antenna.float().to(device)

    if model is None:
        model = DeepResField(Block, [2, 3, 1]).to(device)
        if load:
            checkpoint = torch.load('checkpoint.pt.tar')
            model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam([input_antenna.requires_grad_()], lr=0.0001, weight_decay=0.001)

    for step in range(numsteps):

        output = model(input_antenna)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        input_antenna.data.clamp_(0., 1.)  # input_antenna = torch.clamp(input_antenna, min=0., max=1.)

        if step % 500 == 0:
            print("Step: {}, loss: {}, input: {}".format(step, loss.item(), input_antenna))

    final_antenna = torch.where(input_antenna >= torch.Tensor([0.5]).to(device), torch.Tensor([1]).to(device),
                                torch.Tensor([0]).to(device))
    print(final_antenna)
    test.draw_antenna(final_antenna[0], 'final_antenna1.jpg')
    torch.save(final_antenna, 'final_antenna.pt')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #antennas, fields = read_dataset()
    antennas, fields = read_datasets()

    best = test.bestAntennas(antennas, fields)
    #test.plot_field(fields[0])
    #test.draw_antenna(antennas[0], 'first_antenna_correct7.jpg')
    #test.randomness(fields)

    #model = training(antennas, fields)

    shapeOptimizer(best)#, model=model)
