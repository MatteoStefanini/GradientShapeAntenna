import numpy as np
import pandas as pd
import torch
import cv2


def draw_antenna(mat, name):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                mat[i][j] = 255

    dst = cv2.resize(mat, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)
    cv2.imwrite(name, dst)
    cv2.waitKey(0)


def fill_antenna_center():
    ant = np.zeros(504, dtype='uint8')
    antenna = np.insert(ant, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1 , 1, 1, 1, 1, 1])
    antenna = antenna.reshape((32, 16))
    draw_antenna(antenna)


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

    # antennas = read_antenna_dataset()
    field = read_field_dataset()
