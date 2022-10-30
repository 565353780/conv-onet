#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from shutil import copyfile

dataset_folder_path = "/home/chli/chLi/conv-onet/demo_train/"
dataset_name = "00000001"

pointcloud_folder_path = dataset_folder_path + dataset_name + "/pointcloud/"
points_iou_folder_path = dataset_folder_path + dataset_name + "/points_iou/"

pointcloud_file_name_list = os.listdir(pointcloud_folder_path)

for pointcloud_file_name in pointcloud_file_name_list:
    pointcloud_idx = pointcloud_file_name.split(".")[0].split("_")[1]

    pointcloud_file_path = pointcloud_folder_path + "pointcloud_" + pointcloud_idx + ".npz"
    points_iou_file_path = points_iou_folder_path + "points_iou_" + pointcloud_idx + ".npz"

    new_folder_path = dataset_folder_path + pointcloud_idx + "/"
    os.makedirs(new_folder_path, exist_ok=True)

    new_pointcloud_file_path = new_folder_path + "pointcloud.npz"
    new_points_iou_file_path = new_folder_path + "points_iou.npz"

    copyfile(pointcloud_file_path, new_pointcloud_file_path)
    copyfile(points_iou_file_path, new_points_iou_file_path)
