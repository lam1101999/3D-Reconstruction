import numpy as np
import torch
import sys

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
    centroid = np.mean(points, dim=0, keepdim=True)
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
    selected_flag = np.zeros(n_face, dtype=np.bool)
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
    
    vertex_flag = np.ones(n_vertex, stype=np.int32)*-1
    for i, v in enumerate(all_vertex):
        if vertex_flag[v] < 0:
            vertex_flag[v] = len(v_idx)
            f[i] = len(v_idx)
            v_idx.append(v)
        else:
            f[i] = vertex_flag[v]
    return np.array(v_idx), f.reshape(len(select_faces), 3)