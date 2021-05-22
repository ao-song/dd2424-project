import numpy as np
from sklearn.neighbors import NearestNeighbors

def getQ(pixels):
    colors = np.zeros((22, 22))

    for p in pixels:
        a, b = p
        colors[get_index(a), get_index(b)] = 1

    return np.count_nonzero(colors)


def get_index(num):
    return (num + 110) / 10


def get_space():
    # Cifar10 occupied the full color space
    a = np.arange(-110, 110, 10)
    b = np.arange(-110, 110, 10)

    space = []
    for i in a:
        for j in b:
            space.append([i, j])

    return np.array(space)


def gaussian_kernel(distance, sigma=5):
    a = np.exp(-np.power(distance, 2) / (2*np.power(sigma, 2)))
    b = np.sum(a, axis=1).reshape(-1, 1)
    return a / b


def soft_encoding_ab(ab):
    n = ab.shape[0]
    Y = []

    for i in range(n):
        # Flatten the a and b and construct 2d array
        a = ab[i, 0, :, :]
        b = ab[i, 1, :, :]
        # print(a.shape)
        a = a.flatten()
        # print(a.shape)
        b = b.flatten()
        newab = np.vstack((a, b)).T
        # Full color space
        space = get_space()
        # Compute soft encoding
        nbrs = NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(space)
        distances, indices = nbrs.kneighbors(newab)
        # print('indices is: ' + str(indices))
        # print(indices.shape)
        gk = gaussian_kernel(distances)
        # print('gk is : ' + str(gk))
        # print(gk.shape)
        y = np.zeros((newab.shape[0], space.shape[0]))
        # print(y.shape)
        index = np.arange(newab.shape[0]).reshape(-1, 1)
        # print(index)
        y[index, indices] = gk
        # print(y.shape)
        y = y.reshape(ab[i, 0, :, :].shape[0], ab[i, 0, :, :].shape[1], space.shape[0])
        Y.append(y.T)

    return np.stack(Y)
