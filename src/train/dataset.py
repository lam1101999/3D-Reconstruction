import torch_geometric
import sys
import os
import glob
import openmesh as om
import numpy as np
import data_utils
import torch
from torch_geometric.utils import to_undirected, add_self_loops
from tqdm import tqdm
from torch_geometric.data import Data, Batch

class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, tuple):
            return elem

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class RandomRotate(object):
    def __init__(self, z_rotated=True):
        self.z_rotated = z_rotated

    def __call__(self, data):
        # rotation
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = Rz if self.z_rotated else np.dot(Rz, np.dot(Ry, Rx))
        rotation_matrix = torch.from_numpy(rotation_matrix).to(data[0].y.dtype).to(data[0].y.device)

        for d in data:
            d.x[:, 0:3] = torch.matmul(d.x[:, 0:3], rotation_matrix)
            d.x[:, 3:6] = torch.matmul(d.x[:, 3:6], rotation_matrix)
            d.y[:, 0:3] = torch.matmul(d.y[:, 0:3], rotation_matrix)
            if hasattr(d, 'pos') and d.pos is not None:
                d.pos = torch.matmul(d.pos, rotation_matrix)
            if hasattr(d, 'centroid') and d.centroid is not None:
                d.centroid = torch.matmul(d.centroid, rotation_matrix)
            if hasattr(d, 'depth_direction') and d.depth_direction is not None:
                d.depth_direction = torch.matmul(d.depth_direction, rotation_matrix)

        return data

class DualDataset(torch_geometric.data.Dataset):
    def __init__(self, data_type, train_or_test="train",filter_patch_count=0, submesh_size=sys.maxsize, transform=None):
        super().__init__(transform=transform)
        CODE_DIR = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
        DATASET_DIR = os.path.join(BASE_DIR,"dataset")
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
        self.processed_information_file_path = os.path.join(self.processed_dir, "processed_information.pt")
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
        """
        Process the data by creating a processed directory, iterating over the noisy files, and calling the `process_one_data` method for each file.

        Parameters:
            None

        Returns:
            None
        """

        os.makedirs(self.processed_dir, exist_ok=True)
        if os.path.exists(self.processed_information_file_path): # load processed information if exist
            self.processed_files = torch.load(self.processed_information_file_path)
            self.processed_files.sort()
            print("Load processed information!")
        else:
            print("Processing...")
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            pbar = tqdm(total=len(self.noisy_files), ncols=85, desc="Processing Data", bar_format=bar_format)
            for i, (original_file, noisy_file) in enumerate(zip(self.original_files, self.noisy_files)):
                pbar.postfix = os.path.basename(original_file)[:-4]
                pbar.update(1)
                self.process_one_data(noisy_file, self.submesh_size,original_file=original_file, obj=self)
            torch.save(self.processed_files, self.processed_information_file_path)
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
            list: A list of tuples containing the processed data. This is because mesh can be split into smaller submeshes.
                Each tuple contains (noisy data, original data:optional), v_idx:optional, select_face:optional.
        """
        filter_patch_count = 0 if obj is None else obj.filter_patch_count
        
        # 1. load data
        noisy_mesh = om.read_trimesh(noisy_file)
        original_mesh = None if original_file is None else om.read_trimesh(original_file)
        
        noisy_points = noisy_mesh.points().astype(np.float32)
        original_points = None if original_mesh is None else original_mesh.points().astype(np.float32)
        
        # 2. center and scale, record
        _, centroid, scale = data_utils.center_and_scale(noisy_points, noisy_mesh.ev_indices(), 0)
        
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
            flag = np.zeros(noisy_mesh.n_faces(), dtype=bool)
            
            fv_indices =  noisy_mesh.fv_indices()
            vf_indices = noisy_mesh.vf_indices()
            face_center = noisy_points[fv_indices].mean(1)
            seed = np.argmax(((face_center-centroid)**2).sum(1)) # find index of face farthest to centroid
            for sub_num in range(1, sys.maxsize):
                # Select patch facet indices
                select_faces = data_utils.mesh_get_neighbor_np(fv_indices, vf_indices,seed,submesh_size)
                flag.put(select_faces, True)
                
                if len(select_faces) > filter_patch_count:
                    file_name = f"{os.path.basename(noisy_file)[:-4]}-sub{submesh_size}-{seed}"
                    if obj is not None:
                        processed_path = os.path.join(obj.processed_dir, f"{file_name}.pt")
                        obj.processed_files.append(processed_path)
                    if obj is None or not os.path.exists(processed_path):
                        # split submesh based on facet indices
                        v_idx, f = data_utils.get_submesh(fv_indices, select_faces)
                        submesh_n = om.TriMesh(noisy_points[v_idx], f)
                        submesh_o = None if original_points is None else om.TriMesh(original_points[v_idx], f)
                        dual_data = DualDataset.process_one_submesh(submesh_n, file_name, submesh_o)
                        dual_data[0].centroid = torch.from_numpy(centroid).float()
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
    def process_one_data_from_mesh(mesh, submesh_size):
        """
        This function processes one data from the mesh.

        Parameters:
        data (dict): The data to be processed. The data should be a dictionary with the following keys:
            - 'mesh' (TriMesh): the mesh.
            - 'submesh_size' (int): The size of the submesh.

        Returns:
            list: A list of tuples containing the processed data. This is because mesh can be split into smaller submeshes.
                Each tuple contains (noisy data, original data:optional), v_idx:optional, select_face:optional.
        """
        filter_patch_count = 0
        
        # 1. load data
        points = mesh.points().astype(np.float32)
        
        # 2. center and scale, record
        # print(points.shape)
        # print(mesh.ev_indices().shape)
        _, centroid, scale = data_utils.center_and_scale(points, mesh.ev_indices(), 0)
        
        # 3. split to submesh
        all_dual_data = []
        if mesh.n_faces() <= submesh_size:
            dual_data = DualDataset.process_one_submesh(mesh)
            dual_data[0].centroid = torch.from_numpy(centroid).float()
            dual_data[0].scale = scale
            all_dual_data.append((dual_data, None, None))
        else:
            flag = np.zeros(mesh.n_faces(), dtype=bool)
            
            fv_indices =  mesh.fv_indices()
            vf_indices = mesh.vf_indices()
            face_center = points[fv_indices].mean(1)
            seed = np.argmax(((face_center-centroid)**2).sum(1)) # find index of face farthest to centroid
            for sub_num in range(1, sys.maxsize):
                # Select patch facet indices
                select_faces = data_utils.mesh_get_neighbor_np(fv_indices, vf_indices,seed,submesh_size)
                flag.put(select_faces, True)
                
                if len(select_faces) > filter_patch_count:
                    # split submesh based on facet indices
                    v_idx, f = data_utils.get_submesh(fv_indices, select_faces)
                    submesh_n = om.TriMesh(points[v_idx], f)
                    dual_data = DualDataset.process_one_submesh(submesh_n)
                    dual_data[0].centroid = torch.from_numpy(centroid).float()
                    dual_data[0].scale = scale
                    all_dual_data.append((dual_data, v_idx, select_faces))
                # whether all facets of current seed have been visited, next seed
                left_idx = np.where(~flag)[0] # find unvisited faces
                if left_idx.size:
                    idx_temp = np.argmax(((face_center[left_idx] - centroid)**2).sum(1)) # find remaining face farthest to centroid
                    seed = left_idx[idx_temp] # convert it from temp idx to global face index
                else:
                    break
        return all_dual_data   

    @staticmethod
    def process_one_submesh(noisy_mesh, file_name="", original_mesh=None):
        """
        Process one submesh and generate graph data for vertex and facet graphs.
        
        Args:
            noisy_mesh (Mesh): The noisy mesh object.
            file_name (str): The file name of the mesh.
            original_mesh (Mesh, optional): The original mesh object. Defaults to None.
        
        Returns:
            tuple: A tuple containing the graph data for the vertex and facet graphs.
                - graph_vertex (Data): The graph data for the vertex graph.
                - graph_face (Data): The graph data for the facet graph.
        """

        noisy_mesh.update_face_normals()
        noisy_mesh.update_vertex_normals()
        
        ev_indices = torch.from_numpy(noisy_mesh.ev_indices()).long()
        fv_indices = torch.from_numpy(noisy_mesh.fv_indices()).long()
        vf_indices = torch.from_numpy(noisy_mesh.vf_indices()).long()
        
        edge_dual_fv = data_utils.build_edge_fv(fv_indices)
        
        # vertex graph
        position_v = torch.from_numpy(noisy_mesh.points()).float()
        normal_v = torch.from_numpy(noisy_mesh.vertex_normals()).float()
        normal_v = normal_v if len(normal_v.size())==2 else normal_v.unsqueeze(0)
        edge_idx_v = ev_indices.T # directed graph, no self_loops
        edge_idx_v = to_undirected(edge_idx_v)
        edge_idx_v,_ = add_self_loops(edge_idx_v)

        edge_weight_v = data_utils.calc_weight(position_v, normal_v, edge_idx_v).float()
        depth_direction = torch.nn.functional.normalize(position_v, dim=1).float() # Unit vector 
        graph_vertex = Data(pos=position_v, normal = normal_v, edge_index=edge_idx_v, edge_weight = edge_weight_v,
                            depth_direction = depth_direction, edge_dual = edge_dual_fv[1])
        
        # face graph
        position_f = position_v[fv_indices].mean(1).float()
        normal_f = torch.from_numpy(noisy_mesh.face_normals()).float()
        normal_f = normal_f if len(normal_f.size())==2 else normal_f.unsqueeze(0)
        edge_idx_f = data_utils.build_facet_graph(fv_indices, vf_indices) # undirected graph, with self loops
        edge_weight_f = data_utils.calc_weight(position_f, normal_f, edge_idx_f).float()
        graph_face = Data(pos=position_f, normal = normal_f, edge_index=edge_idx_f, edge_weight = edge_weight_f, fv_indices=fv_indices, edge_dual=edge_dual_fv[0])
        
        if original_mesh is not None:
            original_mesh.update_vertex_normals()
            original_mesh.update_face_normals()
            # Vertex graph
            graph_vertex.y = torch.from_numpy(original_mesh.points()).float()
            # facet graph
            graph_face.y = torch.from_numpy(original_mesh.face_normals()).float()
        
        return (graph_vertex, graph_face)
    @staticmethod
    def post_processing(dual_data, is_plot=False):
        """
        Post-processes the given dual data.

        Parameters:
            dual_data (tuple): A tuple containing the vertex and face data.
            data_type (str): The type of data being processed.
            is_plot (bool, optional): Whether or not the data is meant for plotting. Defaults to False.

        Returns:
            tuple: A tuple containing the processed vertex and face data.
        """
        data_vertex, data_face = dual_data
        
        # Facet graph
        position_facet_normalized = (data_face.pos - data_vertex.centroid) * data_vertex.scale
        data_face.x = torch.cat((position_facet_normalized, data_face.normal), 1) # normalized average position vertex of face, normal vector of facet
        data_face.normal = data_face.edge_dual = None
        
        # vertex graph
        position_vertex_normalized = (data_vertex.pos - data_vertex.centroid) * data_vertex.scale
        data_vertex.x = torch.cat((position_vertex_normalized, data_vertex.normal), 1) # normalized position of vertex, normal vector of vertex
        data_vertex.y = None if data_vertex.y is None else (data_vertex.y - data_vertex.centroid)*data_vertex.scale
        data_vertex.normal = data_vertex.centroid = data_vertex.scale = data_vertex.edge_dual = None
        


        if not is_plot:
            data_vertex.pos = None
            data_face.pos = None
        else:
            data_vertex.pos = data_vertex.y
            data_vertex.fv_indices = data_face.fv_indices
            data_vertex.depth_direction = None
        return data_vertex, data_face
    def len(self):
        return len(self.processed_files)
    def get(self, idx):
        dual_data = torch.load(self.processed_files[idx])
        return DualDataset.post_processing(dual_data)
    