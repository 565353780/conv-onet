import os
import numpy as np
import open3d as o3d
from time import time

from conv_onet.Module.detector import Detector


if __name__ == "__main__":
    pcd_folder_path = "/home/chli/chLi/Dataset/ShapeNet/manifold_pcd-8192_random-10/"
    sample_point_num = 8192
    save_folder_path = "./output/ShapeNet/"

    if not os.path.exists(pcd_folder_path):
        print('folder not exist!')
        exit()

    detector = Detector()

    os.makedirs(save_folder_path, exist_ok=True)

    i = 0
    for root, _, files in os.walk(pcd_folder_path):
        for file in files:
            if not file.endswith('.ply'):
                continue

            pcd_file_path = root + '/' + file

            start = time()
            pcd = o3d.io.read_point_cloud(pcd_file_path)

            sampled_pcd = pcd
            if len(pcd.points) > sample_point_num:
                sampled_pcd = pcd.farthest_point_down_sample(sample_point_num)

            points = np.asarray(sampled_pcd.points).reshape(1, -1, 3)

            detector.detectAndSave(points, save_folder_path + 'mesh_' + str(i) + '.obj', False, True)

            time_spend = time() - start
            i += 1
            print('finish convert', i, '! time:', time_spend)
