import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter


def discount(x, gamma):
    return lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def permute(list_data):
    permutation = np.random.permutation(list_data[0].shape[0])
    for data in list_data:
        data[...] = data[permutation]
    return list_data


def GAE(lamb, td_errors, lengths):
    gaes = []
    first = 0
    for T in lengths:
        last = first + T
        lambda_pows = toeplitz(lamb ** np.arange(T),
                               np.zeros(T))
        gaes.append(np.dot(lambda_pows, td_errors[first:last]))
        first = last
    return np.concatenate(gaes)
