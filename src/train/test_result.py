# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse
import numpy as np
import openmesh as om
import torch
import data_utils as data_utils
from datetime import datetime
from typing import Union
from dataset import DualDataset

from loss import error_n, error_v


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and sys.gettrace()


def predict_one_submesh(net, device, dual_data):
    """
    A function that predicts one submesh using the given neural network and input data.

    Args:
        net: The neural network model.
        device: The device on which to run the prediction.
        dual_data: The input data for the prediction.

    Returns:
        vert_p: Predicted vertices.
        norm_p: Predicted normals.
    """
    with torch.no_grad():
        dual_data = [d.to(device) for d in dual_data]
        vert_p, norm_p, alpha = net(dual_data)
        return vert_p, norm_p

def inference(filename, net, device:Union[str,torch.device]="cpu", sub_size=20000, data_type=None):
    """
    Perform inference on a given mesh using the provided neural network.

    Args:
        filename (str): The file path of the mesh to be processed.
        net: The neural network model to be used for inference.
        device: The device on which the inference will be performed.
        sub_size: The size of the submesh for processing.
        data_type (optional): The type of data used for post-processing.

    Returns:
        om.TriMesh: The updated mesh after inference.
        Np: The updated normals of the mesh after inference.
    """
    # 1.load data
    mesh_noisy = om.read_trimesh(filename)
    points_noisy = mesh_noisy.points().astype(np.float32)

    # 2.process entire mesh
    all_data = DualDataset.process_one_data(
        filename, sub_size)
    centroid = all_data[0][0][0].centroid
    scale = all_data[0][0][0].scale

    # 3.inference
    if len(all_data) == 1:
        dual_data = all_data[0][0]
        dual_data = DualDataset.post_processing(dual_data, data_type)
        Vp, Np = predict_one_submesh(net, device, dual_data)
    else:
        sum_v = torch.zeros((mesh_noisy.n_vertices(), 1),
                            dtype=torch.int8, device=device)
        Vp = torch.zeros((mesh_noisy.n_vertices(), 3),
                         dtype=torch.float32, device=device)
        Np = torch.zeros((mesh_noisy.n_faces(), 3),
                         dtype=torch.float32, device=device)

        for dual_data, V_idx, F_idx in all_data:
            dual_data = DualDataset.post_processing(dual_data, data_type)
            vert_p, norm_p = predict_one_submesh(net, device, dual_data)
            sum_v[V_idx] += 1
            Vp[V_idx] += vert_p
            Np[F_idx] += norm_p

        Vp /= sum_v
        Np = torch.nn.functional.normalize(Np, dim=1)

    Vp = Vp.cpu() / scale + centroid
    Np = Np.cpu()

    # 4.update position and save
    fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
    vf_indices = torch.from_numpy(mesh_noisy.vf_indices()).long()
    depth_direction = None
    if data_type in ['Kinect_v1', 'Kinect_v2']:
        depth_direction = torch.nn.functional.normalize(
            torch.from_numpy(points_noisy), dim=1)
    V = data_utils.update_position2(
        Vp, fv_indices, vf_indices, Np, 60, depth_direction=depth_direction)
    return om.TriMesh(V.numpy(), mesh_noisy.fv_indices()), Np

def evaluate_one_mesh(filename, net, device, sub_size, rst_filename=None, filename_gt=None):
    """
    Evaluate a mesh using a neural network model.

    Parameters:
    - filename: str, the file name of the mesh to be evaluated
    - net: neural network model, the model used for evaluation
    - device: str, the device on which the evaluation will be performed
    - sub_size: int, the size of the sub-meshes for evaluation
    - rst_filename: str, optional, the file name for the resulting mesh
    - filename_gt: str, optional, the file name of the ground truth mesh for comparison

    Returns:
    - angle1: float, the error angle between the predicted normals and the ground truth normals
    - angle2: float, the error angle between the computed normals and the ground truth normals
    - Np.shape[0]: int, the number of faces in the denoised mesh
    - vertice_distance: float, the error distance between the predicted vertices and the ground truth vertices
    - V.shape[0]: int, the number of vertices in the denoised mesh
    """
    from loss import error_n, error_v

    denoised_mesh, Np = inference(filename, net, device, sub_size)
    V = torch.tensor(denoised_mesh.points().astype(float))
    fv_indices = torch.from_numpy(denoised_mesh.fv_indices()).long()
    if rst_filename is not None:
        om.write_mesh(F"{rst_filename[:-4]}-60.obj",
                  denoised_mesh)
    

    angle1 = angle2 = 0
    if filename_gt is not None:
        mesh_o = om.read_trimesh(filename_gt)
        mesh_o.update_face_normals()
        Nt = torch.from_numpy(mesh_o.face_normals()).float()
        angle1 = error_n(Np, Nt)
        Np2 = data_utils.compute_face_normal(V, fv_indices)
        angle2 = error_n(Np2, Nt)
        vertice_distance = error_v(V, torch.from_numpy(mesh_o.points().astype(float)).float())
    return angle1, angle2, Np.shape[0], vertice_distance, V.shape[0]

def evaluate_two_dir(original_data_dir, denoised_data_dir):

    log_file = os.path.join(denoised_data_dir,"log_evaluate_two_dir.txt")
    filenames = []
    filenames_gt = []
    data_list = glob.glob(os.path.join(original_data_dir, '*.obj'))
    data_names = [os.path.basename(original_file)[:-4] for original_file in data_list]
    for data_name in data_names:
        noisy_files = glob.glob(os.path.join(denoised_data_dir, f"{data_name}_n*.obj"))
        for noisy_file in noisy_files:
            filenames.append(noisy_file)
            filenames_gt.append(os.path.join(original_data_dir, f"{data_name}.obj"))

    error_all = np.zeros((4, len(filenames)))  # count, mean_error
    for i,(filename, filename_gt) in enumerate(zip(filenames, filenames_gt)):
        original_mesh = om.read_trimesh(filename_gt)
        denoised_mesh = om.read_trimesh(filename)
        original_mesh.update_face_normals()
        denoised_mesh.update_face_normals()

        normal_original_mesh = torch.from_numpy(original_mesh.face_normals()).float()
        normal_denoised_mesh = torch.from_numpy(denoised_mesh.face_normals()).float()
        num_faces = normal_original_mesh.shape[0]
        angle = error_n(normal_original_mesh, normal_denoised_mesh)

        vertice_original_mesh = torch.from_numpy(original_mesh.points().astype(float))
        vertice_denoised_mesh = torch.from_numpy(denoised_mesh.points().astype(float))
        num_vertices = vertice_original_mesh.shape[0]
        vertice_distance = error_v(vertice_original_mesh, vertice_denoised_mesh)

        error_all[0, i] = num_faces
        error_all[1, i] = angle
        error_all[2, i] = vertice_original_mesh.shape[0]
        error_all[3, i] = vertice_distance
        result_one_mesh = (F"\nangle: {angle:9.6f},  faces: {num_faces:>6}, vertice_distance: {vertice_distance:9.6f}, vertices:{num_vertices}, '{os.path.basename(filename)}'")

        print(result_one_mesh)
        f = open(f"{log_file}", "a")
        f.write(result_one_mesh)
        f.close()

    count_faces = error_all[0].sum().astype(np.int32)
    error_mean = (error_all[0]*error_all[1]).sum() / count_faces
    count_vertices = error_all[2].sum().astype(int)
    error_vertice_mean = (error_all[2]*error_all[3]).sum() / count_vertices
    result = (f"\nNum_face: {count_faces:>6},  angle_mean: {error_mean:.6f}, Num_vertice: {count_vertices}, vertice_mean: {error_vertice_mean:.6f} ")

    print(result)
    f = open(f"{log_file}", "a")
    f.write(result)
    f.close()

def evaluate_dir(params_path, data_dir=None, sub_size=None, gpu=-1, write_to_file=False):
    """
    A function to evaluate a directory of data using a given set of parameters.

    :param params_path: The path to the parameters file.
    :param data_dir: The directory containing the data to be evaluated. Default is None.
    :param sub_size: The size of the sub-data. Default is None.
    :param gpu: The GPU index. Default is -1.
    :param write_to_file: Whether to write the results to a file. Default is False.
    """
    assert (data_dir is None or os.path.exists(data_dir))

    opt = torch.load(params_path)
    opt.sub_size = opt.sub_size if sub_size is None else sub_size
    print('\n' + str(opt) + '\n')
    bak_dir = os.path.dirname(params_path)

    # 1. prepare data
    filenames = []
    filenames_gt = []
    if data_dir is None:
        data_dir = os.path.join(DATASET_DIR, opt.data_type, 'test')
    original_dir = os.path.join(data_dir, 'original')
    data_list = glob.glob(os.path.join(original_dir, "*.obj"))
    data_list = [os.path.basename(d)[:-4] for d in data_list]
    for name in data_list:
        files_n = glob.glob(os.path.join(
            data_dir, 'noisy', F"{name}_n*.obj"))
        for name_n in files_n:
            filenames.append(name_n)
            filenames_gt.append(os.path.join(original_dir, F"{name}.obj"))      
    print(
        F"\nEvaluate {opt.flag}, sub_size:{opt.sub_size}, {len(filenames)} files ...\n")
    result_dir = os.path.join(data_dir, F"result_{opt.flag}")
    log_file = os.path.join(result_dir,"log.txt")
    os.makedirs(result_dir, exist_ok=True)

    # 2. model
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif gpu >= 0:
        device = torch.device(F"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    sys.path.insert(0, bak_dir)
    import network
    net = network.DualGenerator(force_depth=opt.force_depth,
                          pool_type=opt.pool_type, wei_param=opt.wei_param)
    net.load_state_dict(torch.load(os.path.join(bak_dir, opt.model_name)))
    net = net.to(device)
    net.eval()

    # 3. infer
    error_all = np.zeros((5, len(filenames)))  # count, mean_error
    for i, noisy_file in enumerate(filenames):
        rst_file = os.path.join(result_dir, os.path.basename(noisy_file))
        # if os.path.exists(rst_file):
        #     continue
        angle1, angle2, num_faces, vertice_distance, num_vertice = evaluate_one_mesh(noisy_file, net, device, sub_size, rst_file, None if len(
            filenames_gt) == 0 else filenames_gt[i])
        error_all[0, i] = num_faces
        error_all[1, i] = angle1
        error_all[2, i] = angle2
        error_all[3, i] = num_vertice
        error_all[4, i] = vertice_distance
        result_one_mesh = (F"\nangle1: {angle1:9.6f},  angle2: {angle2:9.6f},  faces: {num_faces:>6}, vertice_distance: {vertice_distance:9.6f}, vertices:{num_vertice}, '{os.path.basename(noisy_file)}'")
        print(result_one_mesh)
        if write_to_file:
            f = open(f"{log_file}", "a")
            f.write(result_one_mesh)
            f.close()

    count_faces = error_all[0].sum().astype(np.int32)
    error_mean1 = (error_all[0]*error_all[1]).sum() / count_faces
    error_mean2 = (error_all[0]*error_all[2]).sum() / count_faces
    count_vertices = error_all[3].sum().astype(int)
    error_vertice_mean = (error_all[3]*error_all[4]).sum() / count_vertices
    result = (f"\nNum_face: {count_faces:>6},  angle_mean1: {error_mean1:.6f},  angle_mean2: {error_mean2:.6f}, Num_vertice: {count_vertices}, vertice_mean: {error_vertice_mean:.6f} ")
    print(result)
    if write_to_file:
        f = open(f"{log_file}", "a")
        f.write(result)
        f.close()
    print("\n--- end ---")

def denoise_dir(model_path, data_dir, sub_size=20000, force_depth=False, pool_type='max', wei_param=2):
    """
    Denoise the files in the given directory using the specified model and parameters.

    Args:
        model_path (str): The path to the denoising model.
        data_dir (str): The directory containing the input noisy files.
        sub_size (int, optional): The size of the subset of the files to process at a time. Defaults to 20000.
        force_depth (bool, optional): Whether to force the depth during denoising. Defaults to False.
        pool_type (str, optional): The type of pooling to be used. Defaults to 'max'.
        wei_param (int, optional): The weight parameter. Defaults to 2.

    Returns:
        None
    """
    # 1. prepare data
    filenames = glob.glob(os.path.join(data_dir, '*.obj'))

    print(F"\Denoise , sub_size:{sub_size}, {len(filenames)} files ...\n")
    result_dir = data_dir + "_result"
    os.makedirs(result_dir, exist_ok=True)

    # 2. model
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    import network
    net = network.DualGenerator(force_depth=force_depth,
                          pool_type=pool_type, wei_param=wei_param)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    # 3. infer each noisy file
    for i, noisy_file in enumerate(filenames):
        rst_file = os.path.join(result_dir, os.path.basename(noisy_file))
        denoised_mesh, _ = inference(noisy_file, net, device, sub_size)
        om.write_mesh(rst_file, denoised_mesh)
        print(f"finish denoise {noisy_file}")

if __name__ == "__main__":
    IS_DEBUG = getattr(sys,"gettrace", None) is not None and sys.gettrace()
    parser = argparse.ArgumentParser()
    if IS_DEBUG:
        parser.add_argument('--params_path', type=str, default=r"D:\project\3D-Reconstruction\src\log\GeoBi-GNN_Synthetic_train\train_gan\GeoBi-GNN_Synthetic_params.pth", help='params_path')

        parser.add_argument('--data_dir', type=str, default=r"G:\My Drive\data", help='data_dir')
    else:
        parser.add_argument('--params_path', type=str,
                            default=None, help='params_path')
        parser.add_argument('--data_dir', type=str, default=None, help='data_dir')
    parser.add_argument('--sub_size', type=int,
                        default=20000, help='submesh size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    opt = parser.parse_args()
    print("-------------------------------")
    print(opt)

    # evaluate_dir(opt.params_path, data_dir=opt.data_dir,sub_size=opt.sub_size, gpu=opt.gpu, write_to_file=True)
    evaluate_two_dir(r"G:\My Drive\data\original",r"G:\My Drive\data\denoise_ours")

