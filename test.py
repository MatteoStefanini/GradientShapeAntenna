import numpy as np
import pandas as pd
import torch
import cv2


def draw_antenna(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                mat[i][j] = 255

    cv2.imwrite('antenna.png', mat)
    cv2.waitKey(0)


def fill_antenna():
    ant = np.zeros(504, dtype='uint8')
    antenna = np.insert(ant, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1 , 1, 1, 1, 1, 1])
    antenna = antenna.reshape((32, 16))
    draw_antenna(antenna)


if __name__ == '__main__':

    #fill_antenna()

    #dt = np.dtype([('combo', 'uint8', (510,)), ('fitness', 'float32')])
    #records = np.fromfile('dataset_210618_1644.dat', dt)
    dt = np.dtype([('combination', 'uint8', (504,))])
    records = np.fromfile('dataset_pop_260618_1642.dat', dt)
    print(records.shape)
    data = np.concatenate(records.tolist(), axis=0)
    print(data.shape)

    #df = pd.DataFrame(data, columns=records.dtype.names)
    #print(df.shape)
    # df = torch.from_numpy(records)

    antennas = np.insert(data, [224, 224, 238, 238, 252, 252, 266, 266], [1, 1, 1, 1, 1, 1, 1, 1], axis=1)\
        .reshape((data.shape[0], 32, 16))
    print(antennas.shape)

    #df2 = pd.read_csv('dataset_210618_1301.dat', header=None)
    #print(df2.shape)
