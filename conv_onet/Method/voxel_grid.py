#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage

from src.utils.libvoxelize.voxelize import voxelize_mesh_
from src.utils.libmesh import check_mesh_contains

from conv_onet.Method.common import make_3d_grid


def voxelize_surface(mesh, resolution):
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = (vertices + 0.5) * resolution

    face_loc = vertices[faces]
    occ = np.full((resolution, ) * 3, 0, dtype=np.int32)
    face_loc = face_loc.astype(np.float32)

    voxelize_mesh_(occ, face_loc)
    occ = (occ != 0)

    return occ


def voxelize_interior(mesh, resolution):
    shape = (resolution, ) * 3
    bb_min = (0.5, ) * 3
    bb_max = (resolution - 0.5, ) * 3
    # Create points. Add noise to break symmetry
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
    points = (points / resolution - 0.5)
    occ = check_mesh_contains(mesh, points)
    occ = occ.reshape(shape)
    return occ


def voxelize_ray(mesh, resolution):
    occ_surface = voxelize_surface(mesh, resolution)
    # TODO: use surface voxels here?
    occ_interior = voxelize_interior(mesh, resolution)
    occ = (occ_interior | occ_surface)
    return occ


def voxelize_fill(mesh, resolution):
    bounds = mesh.bounds
    if (np.abs(bounds) >= 0.5).any():
        raise ValueError(
            'voxelize fill is only supported if mesh is inside [-0.5, 0.5]^3/')

    occ = voxelize_surface(mesh, resolution)
    occ = ndimage.morphology.binary_fill_holes(occ)
    return occ


def check_voxel_occupied(occupancy_grid):
    occ = occupancy_grid

    occupied = (occ[..., :-1, :-1, :-1]
                & occ[..., :-1, :-1, 1:]
                & occ[..., :-1, 1:, :-1]
                & occ[..., :-1, 1:, 1:]
                & occ[..., 1:, :-1, :-1]
                & occ[..., 1:, :-1, 1:]
                & occ[..., 1:, 1:, :-1]
                & occ[..., 1:, 1:, 1:])
    return occupied


def check_voxel_unoccupied(occupancy_grid):
    occ = occupancy_grid

    unoccupied = ~(occ[..., :-1, :-1, :-1]
                   | occ[..., :-1, :-1, 1:]
                   | occ[..., :-1, 1:, :-1]
                   | occ[..., :-1, 1:, 1:]
                   | occ[..., 1:, :-1, :-1]
                   | occ[..., 1:, :-1, 1:]
                   | occ[..., 1:, 1:, :-1]
                   | occ[..., 1:, 1:, 1:])
    return unoccupied


def check_voxel_boundary(occupancy_grid):
    occupied = check_voxel_occupied(occupancy_grid)
    unoccupied = check_voxel_unoccupied(occupancy_grid)
    return ~occupied & ~unoccupied
