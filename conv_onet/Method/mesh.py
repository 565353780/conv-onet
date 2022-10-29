#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def upsample3d_nn(x):
    xshape = x.shape
    yshape = (2 * xshape[0], 2 * xshape[1], 2 * xshape[2])

    y = np.zeros(yshape, dtype=x.dtype)
    y[::2, ::2, ::2] = x
    y[::2, ::2, 1::2] = x
    y[::2, 1::2, ::2] = x
    y[::2, 1::2, 1::2] = x
    y[1::2, ::2, ::2] = x
    y[1::2, ::2, 1::2] = x
    y[1::2, 1::2, ::2] = x
    y[1::2, 1::2, 1::2] = x

    return y


def get_tetrahedon_volume(points):
    vectors = points[..., :3, :] - points[..., 3:, :]
    volume = 1 / 6 * np.linalg.det(vectors)
    return volume


def sample_tetraheda(tetraheda_points, size):
    N_tetraheda = tetraheda_points.shape[0]
    volume = np.abs(get_tetrahedon_volume(tetraheda_points))
    probs = volume / volume.sum()

    tetraheda_rnd = np.random.choice(range(N_tetraheda), p=probs, size=size)
    tetraheda_rnd_points = tetraheda_points[tetraheda_rnd]
    weights_rnd = np.random.dirichlet([1, 1, 1, 1], size=size)
    weights_rnd = weights_rnd.reshape(size, 4, 1)
    points_rnd = (weights_rnd * tetraheda_rnd_points).sum(axis=1)
    # points_rnd = tetraheda_rnd_points.mean(1)

    return points_rnd
