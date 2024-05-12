import os
import numpy as np
import open3d as o3d

from conv_onet.Module.detector import Detector


def demo():
    dataset_folder_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ShapeNet/03001627/'
    sample_point_num = 4000
    save_folder_path = '/home/chli/chLi/Dataset/ConvONet_Manifold_Recon_' + str(sample_point_num) + '/ShapeNet/03001627/'
    render = False
    print_progress = True

    os.makedirs(save_folder_path, exist_ok=True)

    detector = Detector()

    solved_shape_names = os.listdir(save_folder_path)

    pcd_filename_list = os.listdir(dataset_folder_path)
    pcd_filename_list.sort()

    for i, pcd_filename in enumerate(pcd_filename_list):
        if pcd_filename[-4:] != '.npy':
            continue

        if pcd_filename in solved_shape_names:
            continue

        pcd_file_path = dataset_folder_path + pcd_filename

        points = np.load(pcd_file_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        fps_pcd = pcd.farthest_point_down_sample(sample_point_num)

        fps_pts = np.asarray(fps_pcd.points).reshape(1, -1, 3)

        save_mesh_file_path = save_folder_path + pcd_filename.replace('.npy', '.obj')

        detector.detectAndSave(fps_pts, save_mesh_file_path, render, print_progress)

        print('solved shape num:', i + 1)
    return True
