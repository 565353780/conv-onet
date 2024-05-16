import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from time import sleep

from conv_onet.Module.detector import Detector


def demo():
    print('start convert new data...')

    dataset_folder_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ShapeNet/'
    check_folder_path = '/home/chli/chLi/Dataset/MashV4/ShapeNet/'
    sample_point_num = 4000
    save_folder_path = '/home/chli/chLi/Dataset/ConvONet_Manifold_Recon_' + str(sample_point_num) + '/ShapeNet/'
    render = False
    print_progress = True

    detector = Detector()

    classname_list = os.listdir(dataset_folder_path)
    classname_list.sort()

    model_filename_list_dict = {}
    check_model_filename_list_dict = {}
    solved_model_filename_list_dict = {}
    max_shape_num = 0

    for classname in tqdm(classname_list):
        class_folder_path = dataset_folder_path + classname + '/'

        model_filename_list = os.listdir(class_folder_path)
        model_filename_list.sort()

        max_shape_num = max(max_shape_num, len(model_filename_list))
        model_filename_list_dict[classname] = model_filename_list

        check_model_filename_list_dict[classname] = []

        class_check_folder_path = check_folder_path + classname + '/'
        if not os.path.exists(class_check_folder_path):
            continue

        check_model_filename_list = os.listdir(class_check_folder_path)
        check_model_filename_list.sort()
        check_model_filename_list_dict[classname] = check_model_filename_list

        solved_model_filename_list_dict[classname] = []
        class_save_folder_path = save_folder_path + classname + "/"
        os.makedirs(class_save_folder_path, exist_ok=True)
        solved_model_filename_list = os.listdir(class_save_folder_path)
        solved_model_filename_list.sort()
        solved_model_filename_list_dict[classname] = solved_model_filename_list

    solved_shape_num = 0
    for i in range(max_shape_num):
        for classname, model_filename_list in model_filename_list_dict.items():
            if len(model_filename_list) <= i:
                continue

            model_filename = model_filename_list[i]

            if model_filename[-4:] != '.npy':
                continue

            check_model_filename_list = check_model_filename_list_dict[classname]
            if model_filename not in check_model_filename_list:
                continue

            solved_model_filename_list = solved_model_filename_list_dict[classname]
            if model_filename.replace('.npy', '.obj') in solved_model_filename_list:
                continue

            class_folder_path = dataset_folder_path + classname + '/'
            class_save_folder_path = save_folder_path + classname + "/"

            pcd_file_path = class_folder_path + model_filename

            points = np.load(pcd_file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            fps_pcd = pcd.farthest_point_down_sample(sample_point_num)

            fps_pts = np.asarray(fps_pcd.points).reshape(1, -1, 3)

            save_mesh_file_path = class_save_folder_path + model_filename.replace('.npy', '.obj')

            detector.detectAndSave(fps_pts, save_mesh_file_path, render, print_progress)

            solved_shape_num += 1

            print('category:', classname, 'solved shape num:', solved_shape_num)

    print('convert new data finished!')
    return True

if __name__ == "__main__":
    while True:
        demo()
        sleep(10)
