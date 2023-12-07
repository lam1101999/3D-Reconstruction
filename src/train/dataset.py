import torch_geometric
import sys
import os
import glob
import openmesh as om
import numpy as np
import data_utils
import torch

from tqdm import tqdm

class DualDataset(torch_geometric.data.Dataset):
    def __init__(self, data_type, train_or_test="train",filter_patch_count=0, submesh_size=sys.maxsize, transform=None):
        super().__init__(transform=transform)
        CODE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATASET_DIR = os.path.join(CODE_DIR,"dataset")
        self.data_type = data_type
        self.root_dir = os.path.join(DATASET_DIR, data_type)
        self.data_dir = os.path.join(DATASET_DIR, data_type, train_or_test)
        self.filter_patch_count = filter_patch_count
        self.submesh_size = submesh_size
        self.processed_folder_name = "processed_data"

        # 1. Find data path list
        self.noisy_files = []
        self.original_files = []
        self.processed_files = []
        noisy_dir = os.path.join(self.data_dir, "noisy")
        original_dir = os.path.join(self.data_dir, "original")
        self.original_files, self.noisy_files = self.__get_original_and_noisy_data_list(original_dir, noisy_dir)
        
        # 2. Process Data
        self.process_data()
    @property
    def processed_dir(self):
        return os.path.join(self.data_dir, self.processed_folder_name)
    
    def __get_original_and_noisy_data_list(self, original_dir, noisy_dir):
        """
        Retrieves the list of original and noisy data files.

        Args:
            original_dir (str): The directory path where the original data files are located.
            noisy_dir (str): The directory path where the noisy data files are located.

        Returns:
            tuple: A tuple containing two lists - original_files and noisy_files. 
                original_files contains the paths of the original data files.
                noisy_files contains the paths of the corresponding noisy data files.
        """
        original_files = []
        noisy_files = []
        data_path_list = glob.glob(os.path.join(original_dir,"*.obj"))
        data_name_list = [os.path.basename(d)[:-4] for d in data_path_list]
        for data_name in data_name_list:
            corresponding_noisy = glob.glob(os.path.join(noisy_dir, f"{data_name}_n*.obj"))
            for noisy_name in corresponding_noisy:
                original_files.append(os.path.join(original_dir,f"{data_name}.obj"))
                noisy_files.append(noisy_name)
        return original_files, noisy_files
                
    def process_data(self):
        print("Processing...")
        os.makedirs(self.processed_dir, exist_ok=True)
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        pbar = tqdm(total=len(self.noisy_files), ncols=85, desc="Processing Data", bar_format=bar_format)
        for i, (original_file, noisy_file) in enumerate(zip(self.original_files, self.noisy_files)):
            pbar.postfix = os.path.basename(original_file)[:-4]
            pbar.update(1)
            self.process_one_data(noisy_file, self.submesh_size,original_file=original_file)
        pbar.close()
        print("Done!")
    
    @staticmethod
    def process_one_data(noisy_file, submesh_size, original_file=None, obj=None):
        """
        Process one data.
        
        Args:
            noisy_file (str): The path to the noisy file.
            submesh_size (int): The size of the submesh.
            original_file (str): The path to the original file (default: None).
            obj (object): An object (default: None).
        
        Returns:
            list: A list of tuples containing the processed data. 
                Each tuple contains (noisy data, original data:optional), v_idx:optional, select_face:optional.
        """
        filter_patch_count = 0 if obj is None else obj.filter_patch_count
        
        # 1. load data
        noisy_mesh = om.read_trimesh(noisy_file)
        original_mesh = None if original_file is None else om.read_trimesh(original_file)
        
        noisy_points = noisy_mesh.points().astype(np.float32)
        original_points = None if original_mesh is None else original_mesh.points().astype(np.float32)
        
        # 2. center and scale, record
        _, centroid, scale = data_utils.center_and_scale(noisy_points, noisy_mesh.ev_indices(), 1)
        
        # 3. split to submesh
        all_dual_data = []
        if noisy_mesh.n_faces() <= submesh_size:
            file_name = os.path.basename(noisy_file)[:-4]
            if obj is not None: # if is used to create DualDataset instance, save processed path to obj
                processed_path = os.path.join(obj.processed_dir, f"{file_name}.pt")
                obj.processed_files.append(processed_path)
            if obj is None or not os.path.exists(processed_path): # if is used as static function or data is not processed, processed it
                dual_data = DualDataset.process_one_submesh(noisy_mesh, file_name, original_mesh)
                dual_data[0].centroid = torch.from_numpy(centroid).float()
                dual_data[0].scale = scale
                all_dual_data.append((dual_data, None, None))
            if obj is not None and not os.path.exists(processed_path): # If is used to create DualDataset instance and data is not saved, save it
                torch.save(dual_data, processed_path)
        else:
            flag = np.zeros(noisy_mesh.n_faces(), dtype=np.bool)
            
            fv_indices =  noisy_mesh.fv_indices()
            vf_indices = noisy_mesh.vf_indices()
            face_center = noisy_points[fv_indices].mean(1)
            seed = np.argmax(((face_center-centroid)**2).sum(1)) # find index of face farthest to centroid
            for sub_num in range(1, sys.maxsize):
                # Select patch facet indices
                select_faces = data_utils.mesh_get_neighbor_np(fv_indices, vf_indices,seed,submesh_size)
                flag.put(select_faces, True)
                
                if len(select_faces) > filter_patch_count:
                    file_name = f"{os.path.basename(noisy_file)[:-4]}-sub(submesh_size)-{seed}"
                    if obj is not None:
                        processed_path = os.path.join(obj.processed_dir, f"{file_name}.pt")
                        obj.processed_files.append(processed_path)
                    if obj is None or not os.path.exists(processed_path):
                        # split submesh based on facet indices
                        v_idx, f = data_utils.get_submesh(fv_indices, select_faces)
                        submesh_n = om.TriMesh(noisy_points[v_idx], f)
                        submesh_o = None if original_points is None else om.TriMesh(original_points[v_idx], f)
                        dual_data = DualDataset.process_one_submesh(submesh_n, file_name, submesh_o)
                        dual_data[0].centroid = torch.from_numpy()
                        dual_data[0].scale = scale
                        all_dual_data.append((dual_data, v_idx, select_faces))
                    if obj is not None and not os.path.exists(processed_path):
                        torch.save(dual_data, processed_path)
                        # save for visualization
                        om.write_mesh(os.path.join(obj.processed_dir, f"{file_name}.obj"), submesh_n)
                # whether all facets of current seed have been visited, next seed
                left_idx = np.where(~flag)[0] # find unvisited faces
                if left_idx.size:
                    idx_temp = np.argmax(((face_center[left_idx] - centroid)**2).sum(1)) # find remaining face farthest to centroid
                    seed = left_idx[idx_temp] # convert it from temp idx to global face index
                else:
                    break
        return all_dual_data   

    @staticmethod
    def process_one_submesh(noisy_mesh, file_name, original_mesh=None):
        pass
    def get(self,):
        return 1
    
    def len(self):
        return 1

        