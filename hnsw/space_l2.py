# encoding=utf-8

import numpy as np


def get_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum((vec1 - vec2)**2))


def test():
    # vec1 = [1, 2, 3, 4]
    # vec2 = [1, 2, 3, 4]
    dim = 16
    vec1 = np.random.random(dim)
    vec2 = np.random.random(dim)
    vec1 = np.array(vec1)
    print(vec1**2)
    res = get_distance(np.array(vec1), np.array(vec2))
    print(res)


if __name__ == '__main__':
    test()
