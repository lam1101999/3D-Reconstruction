import numpy as np
import torch
import sys
from torch_sparse import coalesce

def center_and_scale(points: np.array, ev_indices: np.array, s_type=0):
    """
	Centers and scales a set of points based on the specified scaling type.

	Parameters:
	- points: np.array
		A numpy array containing the points to be centered and scaled.
	- ev_indices: np.array
		A numpy array containing the indices of the points that form the edges of the shape.
	- s_type: int, optional
		The scaling type to be used. Defaults to 0.
			- 0: Average edge length
			- 1: Bounding box

	Returns:
	- np.array
		The centered and scaled points.
	- np.array
		The centroid of the points.
	- float
		The scaling factor applied to the points.
	"""
    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid

    if s_type == 0 : # Average edge length
        edge_len = points[ev_indices]
        edge_len = ((edge_len[:,0] - edge_len[:,1])**2).sum(1)**0.5
        scale = edge_len.mean()
    elif s_type == 1: # bounding box
        scale  = ((points.max(0) - points.min(0))**2).sum()**0.5
    scale = 1/scale
    return points*scale, centroid, scale

def mesh_get_neighbor_np(fv_indices, vf_indices, seed_idx, neighbor_count=None, ring_count=None):
    """
    Given a set of face-vertex indices and vertex-face indices, this function retrieves the neighboring faces of from seed face.

    Parameters:
        - fv_indices (numpy.ndarray): An array of shape (n, 3) representing the face-vertex indices.
        - vf_indices (numpy.ndarray): An array of shape (m, k) representing the vertex-face indices.
        - seed_idx (int): The index of the seed face.
        - neighbor_count (int): The maximum number of neighboring faces to retrieve. Defaults to None.
        - ring_count (int): The maximum number of rings to iterate. Defaults to None.

    Returns:
        - neighbor (list): A list of indices representing the neighboring faces.

    Raises:
        - AssertionError: If neither neighbor_count nor ring_count is specified.

    Note:
        - If neighbor_count is not specified, it defaults to sys.maxsize.
        - If ring_count is not specified, it defaults to sys.maxsize.
        - The returned neighbor list may contain fewer than neighbor_count faces if the ring_count is reached before that.
    """
    assert (neighbor_count is not None or ring_count is not None), "Please specify neighbor_count or ring_count"
    neighbor_count = sys.maxsize if neighbor_count is None else neighbor_count
    ring_count = sys.maxsize if ring_count is None else ring_count
    
    n_face = fv_indices.shape[0]
    neighbor = []
    selected_flag = np.zeros(n_face, dtype=bool)
    neighbor.append(seed_idx)
    selected_flag[seed_idx] = True
    
    # Loop over ring
    ok_start, ok_end = 0, len(neighbor)
    for ring_idx in range(ring_count): # seed by ring
        for ok_face in neighbor[ok_start:ok_end]:  # loop over the neighbor faces
            for fv in fv_indices[ok_face]: # iterate vertex of ok face
                for fvf in vf_indices[fv]: # iterate neighbor face of that vertex
                    if fvf < 0:
                        break
                    if not selected_flag[fvf]:
                        neighbor.append(fvf)
                        selected_flag[fvf] = True
                        if len(neighbor) >= neighbor_count:
                            return neighbor
        ok_start, ok_end = ok_end, len(neighbor)
        if ok_start ==ok_end:
            return neighbor
    return neighbor

def get_submesh(fv_indices, select_faces):
    """
    Calculates a submesh based on the given face-vertex indices and selected faces.

    Parameters:
        fv_indices (numpy.ndarray): An array of face-vertex indices.
        select_faces (numpy.ndarray): An array of indices representing the selected faces.

    Returns:
        tuple: A tuple containing two arrays:
            - v_idx (numpy.ndarray): An array of vertex indices. This is the indices of original vertices
            - f (numpy.ndarray): An array of face indices reshaped to have 3 columns. The index of vertex is now based on v_idx not original vertices
    """
    n_vertex = fv_indices.max() + 1
    all_vertex = fv_indices[select_faces].flatten()
    
    v_idx = []
    f = np.zeros_like(all_vertex, dtype=np.int32)
    
    vertex_flag = np.ones(n_vertex, dtype=np.int32)*-1
    for i, v in enumerate(all_vertex):
        if vertex_flag[v] < 0:
            vertex_flag[v] = len(v_idx)
            f[i] = len(v_idx)
            v_idx.append(v)
        else:
            f[i] = vertex_flag[v]
    return np.array(v_idx), f.reshape(len(select_faces), 3)

def build_edge_fv(fv_indices):
    """
    Generate the edge index for the given face-vertex indices.

    Parameters:
        fv_indices (torch.Tensor): The face-vertex indices of shape (num_faces, 3).

    Returns:
        torch.Tensor: The edge index of shape (2, num_faces * 3). the first rows denote the index of face and the second rows denote the index its edge

    """
    num_faces = fv_indices.shape[0]
    
    edge_i, _ = torch.meshgrid(torch.arange(num_faces), torch.arange(3), indexing="ij")
    edge_i = edge_i.flatten()
    edge_j = fv_indices.flatten()
    
    edge_index = torch.stack([edge_i, edge_j], 0)
    return edge_index

def calc_weight(node_position, node_normal, edge_index):
    """
    Calculate the weight of a edge based on its vertex position, vertex normal, and edge index.

    Parameters:
    - node_position (torch.Tensor): The position of the node
    - node_normal (torch.Tensor): The normal of the node
    - edge_index (torch.Tensor): The index of the edge

    Returns:
    - torch.Tensor: The weight of the node
    """
    # check the shape
    node_normal = node_normal if len(node_normal.size())==2 else node_normal.unsqueeze(0)
    eps = 0.001
    edge_len = node_position[edge_index]
    # edge_len = ((edge_len[0] - edge_len[1])**2).sum(1)**0.5 # Warning this code is differnet with original one
    # edge_len_mean = edge_len.mean()
    edge_len = ((edge_len[0] - edge_len[1])**2).sum(1) # Warning this code is differnet with original one
    edge_len_mean = (edge_len**0.5).mean()
    
    normal_pair = node_normal[edge_index]
    dn = (normal_pair[0] * normal_pair[1]).sum(1) # dot product of normal_pair
    dp = (edge_len/(-2*edge_len_mean+1e-12)).exp()
    return torch.clamp(dn,eps)*dp

def build_facet_graph(fv_indices, vf_indices):
    """
    Build the facet graph from the given vertex-face indices.
    
    Args:
        fv_indices (Tensor): The indices of vertices for each face.
        vf_indices (Tensor): The indices of faces for each vertex.
        
    Returns:
        torch.Tensor[2, num_pairs]: The edge indices of the facet graph.
                The first rows denotes the index of face and the second rows denote the index of neighbor face
    """
    
    fv_indices = fv_indices.long()
    vf_indices = vf_indices.long()
    
    num_faces = fv_indices.shape[0]
    num_neighbors = vf_indices.shape[1] *3
    
    edge_i, _ = torch.meshgrid(torch.arange(num_faces), torch.arange(num_neighbors), indexing="ij")
    edge_j = vf_indices[fv_indices,:] # if there are no valid face, it is denoted -1
    edge_i = edge_i.flatten()
    edge_j = edge_j.flatten()
    valid_idx = torch.where(edge_j >= 0)[0] # remove invalid pairs (index==-1)
    edge_index = torch.stack([edge_i[valid_idx], edge_j[valid_idx]], 0) # here the edge_index still include repetitive pairs
    edge_index, _ = coalesce(edge_index, None, num_faces, num_faces)  # remove repetitive pairs
    return edge_index

def compute_face_normal(points, fv_indices):
    """
    Compute the face normal of a 3D object given the points and face vertex indices.

    Parameters:
        points (ndarray): The array of points representing the 3D object. Shape: (num_points x 3).
        fv_indices (ndarray): The array of face vertex indices. Shape: (num_faces x 3).

    Returns:
        ndarray or Tensor: The computed face normals. If `fv_indices` is an ndarray, the return type is ndarray with shape (num_faces x 3). If `fv_indices` is a Tensor, the return type is Tensor with shape (num_faces x 3).
    """
    
    fv = points[fv_indices] # (num_faces x 3 x 3)
    if isinstance(fv, np.ndarray):
        N = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0]) # Cross product is perpendicular to the plane (num_faces x 3)
        d = np.clip((N**2).sum(1, keepdims=True)**0.5, 1e-12, None)
        N /= d
    elif isinstance(fv, torch.Tensor):
        N = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
        N = torch.nn.functional.normalize(N, dim=1)
    return N

def update_position2(points, fv_indices, vf_indices, face_normals, n_iter=20, depth_direction=None):
    """
    points: Nx3
    fv_indices: Fx3
    vf_indices: NxNone
    face_normals: Fx3
    """
    if fv_indices.dtype != torch.long:
        fv_indices = fv_indices.long()
    if vf_indices.dtype != torch.long:
        vf_indices = vf_indices.long()

    v_adj_num = (vf_indices > -1).sum(-1, keepdim=True)
    v_adj_num = torch.clamp(v_adj_num, min=1)  # for isolated vertex
    face_normals = torch.cat((face_normals, torch.zeros((1, 3)).to(face_normals.dtype).to(face_normals.device)))
    adj_face_normals = face_normals[vf_indices]

    for _ in range(n_iter):
        face_cent = points[fv_indices].mean(1)
        v_cx = face_cent[vf_indices] - torch.unsqueeze(points, 1)
        d_per_face = (adj_face_normals*v_cx).sum(-1, keepdim=True)
        v_per_face = adj_face_normals * d_per_face
        v_face_mean = v_per_face.sum(1) / v_adj_num
        if depth_direction is not None:  # for Kinect_v1 Kinect_v2 data
            v_face_mean = (v_face_mean * depth_direction).sum(1, keepdim=True)
            v_face_mean = v_face_mean * depth_direction
        points = points + v_face_mean
    return points