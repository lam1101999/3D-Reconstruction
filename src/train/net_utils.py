from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

def pool_edge(cluster, edge_index, edge_attr=None, op="mean"):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op=op)
    return edge_index, edge_attr


def pool_face(cluster, fv_indices):
    face = cluster[fv_indices.view(-1)].view(-1, 3)
    invalid_flag = (face[:,0] == face[:,1]) | (face[:,0] == face[:,2]) | (face[:,1] == face[:,2])
    face = face[~invalid_flag]
    return face
