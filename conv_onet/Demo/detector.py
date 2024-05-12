import numpy as np
import open3d as o3d

from conv_onet.Module.detector import Detector


def demo():
    pcd_file_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ShapeNet/03001627/46bd3baefe788d166c05d60b45815.npy'
    sample_point_num = 4000
    save_mesh_file_path = './output/test.obj'
    render = False
    print_progress = True

    detector = Detector()

    points = np.load(pcd_file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    fps_pcd = pcd.farthest_point_down_sample(sample_point_num)

    fps_pts = np.asarray(fps_pcd.points).reshape(1, -1, 3)

    detector.detectAndSave(fps_pts, save_mesh_file_path, render, print_progress)
    return True
