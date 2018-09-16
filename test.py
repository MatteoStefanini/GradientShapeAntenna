import numpy as np
import pandas as pd
import torch
import cv2


def draw_antenna(mat, name):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                mat[i][j] = 0
            else:
                mat[i][j] = 255

    dst = cv2.resize(mat, None, fx=8, fy=8, interpolation=cv2.INTER_AREA)
    cv2.imwrite(name, dst)
    cv2.waitKey(0)


def plot_field(field):
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d

    Phi = np.radians(np.linspace(-90, 90, 19))
    Theta = np.radians(np.linspace(1, 90, 9))
    THETA, PHI = np.meshgrid(Phi, Theta)

    R = field.reshape((len(Theta), len(Phi)))

    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'),  # or jet
        linewidth=0.1, antialiased=False, alpha=0.5)

    #plt.savefig('field.png', bbox_inches="tight", dpi=300)
    plt.show()


def fill_antenna_center():
    ant = np.zeros(504, dtype='uint8')
    antenna = np.insert(ant, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1 , 1, 1, 1, 1, 1])
    antenna = antenna.reshape((32, 16))
    draw_antenna(antenna)


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


def read_antenna_dataset():
    dt = np.dtype([('combination', 'uint8', (504,))])
    records = np.fromfile('dataset_pop_260618_1642.dat', dt)
    print(records.shape)
    data = np.concatenate(records.tolist(), axis=0)
    print(data.shape)
    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1) \
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)
    return antennas


def read_field_dataset():
    dt = np.dtype([('field', 'float32', (40,))])
    records = np.fromfile('dataset_FF_060718_1928.dat', dt)
    print(records.shape)
    data = np.concatenate(records.tolist(), axis=0)
    print(data.shape)
    return data


if __name__ == '__main__':

    ''' # First tried dataset one file
    dt = np.dtype([('combo', 'uint8', (510,)), ('fitness', 'float32')])
    records = np.fromfile('dataset_210618_1644.dat', dt)
    df = pd.DataFrame(data, columns=records.dtype.names)
    print(df.shape)
    df = torch.from_numpy(records)
    df2 = pd.read_csv('dataset_210618_1301.dat', header=None)
    print(df2.shape)
    '''
    # fill_antenna_center()

    # antennas, fitness = read_dataset_onefitness()

    # antennas = read_antenna_dataset()
    field = read_field_dataset()
