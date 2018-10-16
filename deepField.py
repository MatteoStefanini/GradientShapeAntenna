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
    field = np.concatenate((ftmp1, ftmp2), axis=0)
    print(field.shape)

    return antennas, field


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
    schedulerStep = optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.5)

    for epoch in range(40):
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
    target = torch.Tensor([60,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,16.979443,16.390831,14.699802,12.192764,9.399569,6.9013877,5.045834,3.8627558,3.2152474,2.9814982,3.1461353,3.8298466,5.2948112,7.8841605,11.79757,16.742212,21.767256,25.545385,26.94979,6.4181437,6.523895,6.498592,5.8930774,4.950045,4.247282,3.8436182,3.4621933,3.0293825,2.5091858,1.7981352,1.0051274,0.4590171,0.66554624,2.3632925,5.8081045,9.983548,13.243057,14.445404,1.3993635,1.8733073,2.7961833,3.3903077,3.7685337,4.2237897,4.0460405,3.223515,2.865524,2.9499974,2.5312657,1.6206126,0.8758494,0.29405886,0.291434,1.5794089,3.3652325,4.543933,4.9136353,0.102788955,0.32015356,0.7961047,1.2725159,2.0362575,2.9387867,2.351526,0.94538224,0.7399693,1.2234489,1.239646,0.75646865,0.74405736,0.61022913,0.11292463,0.43044522,0.9508854,1.1471696,1.2036313,0.09539703,0.12557153,0.12608986,0.2644683,0.989888,1.9665514,1.2226223,0.14903641,0.3315221,0.33420357,0.33557105,0.18458776,0.60817224,0.7911495,0.095224656,0.11555943,0.23373608,0.25539184,0.26249105,0.10886455,0.18110645,0.07076928,0.06448294,0.62088424,1.4325254,0.6867487,0.17095229,0.7717876,0.22641554,0.09411606,0.05425589,0.5217974,0.7917129,0.0973483,0.11973389,0.12372471,0.1419548,0.061164614,0.05867293,0.20375477,0.11511716,0.114179164,0.5441823,1.1969062,0.5634693,0.23218735,0.9627118,0.2781146,0.112248875,0.12875688,0.5358263,0.69452596,0.06869823,0.21517006,0.20080702,0.22302145,0.05867293]).to(device)
    target = target.expand(input_antenna.size(0), 153)
    # input_antenna = random
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
    torch.save(final_antenna, 'final_antenna.pt')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    antennas, fields = read_dataset()

    #test.plot_field(fields[0])
    #test.draw_antenna(antennas[0], 'first_antenna_correct7.jpg')
    #test.randomness(fields)

    best = test.bestAntennas(antennas, fields)

    model = training(antennas, fields)

    shapeOptimizer(best, model=model)
