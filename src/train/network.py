import torch
import torch_geometric
import math
from torch_scatter import scatter
from torch.nn import Linear, Parameter, init
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GCNConv, FeaStConv, graclus
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos
try:
    from pytorch3d.ops.points_alignment import iterative_closest_point as icp
except ImportError:
    icp = None
import data_utils
import net_utils
class DualGNN(torch.nn.Module):
    def __init__(self, force_depth=False, pool_type="max", edge_weight_type=10, wei_param=2):
        super().__init__()
        self.force_depth = force_depth
        
        # vertex graph
        self.gnn_v = GNNModule(6, pool_type, 2, edge_weight_type, wei_param) #outchanneels = 32
        self.fc_v1 = torch.nn.Linear(32, 1024)
        self.fc_v2 = torch.nn.Linear(1024, 1) if self.force_depth else torch.nn.Linear(1024, 3)
        
        # facet graph
        self.gnn_f = GNNModule(12, pool_type, 2, edge_weight_type, wei_param) #outchanneels = 32
        self.fc_f1 = torch.nn.Linear(32, 1024)
        self.fc_f2 = torch.nn.Linear(1024,3)
        
    def forward(self, dual_data):
        data_vertex, data_face = dual_data
        xyz = data_vertex.x[:,:3]
        nf = data_face.x[:,3:6]
        
        # Calculate vertex features
        feat_v = self.gnn_v(data_vertex)
        feat_v_h = torch.nn.functional.leaky_relu(self.fc_v1(feat_v), 0.2, inplace=True)
        feat_v = self.fc_v2(feat_v_h)
        if self.force_depth: # Constraint coordinate in the depth direction of original coordiante frame
            feat_v = feat_v * data_vertex.depth_direction
        feat_v += xyz
        
        # Calculate Facet feature
        face_cent = feat_v[data_face.fv_indices].mean(1)
        face_norm = data_utils.compute_face_normal(feat_v, data_face.fv_indices)
        data_face.x = torch.cat((data_face.x, face_cent, face_norm), 1)
        
        feat_f = self.gnn_f(data_face)
        feat_f_h = torch.nn.functional.leaky_relu(self.fc_f1(feat_f), 0.2, inplace=True)
        feat_f = self.fc_f2(feat_f_h)
        
        return feat_v, torch.nn.functional.normalize(feat_f, dim=1), None

class GNNModule(torch.nn.Module):
    def __init__(self, in_channels=6, pool_type="max", pool_step=2, edge_weight_type=0, wei_param=2):
        super().__init__()
        
        self.l_conv1 = FeaStConv(in_channels, 32, 9)
        self.pooling1 = PoolingLayer(32, pool_type, pool_step, edge_weight_type, wei_param)
        self.l_conv2 = FeaStConv(32, 64, 9)
        self.pooling2 =PoolingLayer(64, pool_type, pool_step, edge_weight_type, wei_param)
        self.l_conv3 = FeaStConv(64, 128, 9)
        self.l_conv4 = FeaStConv(128, 128, 9)

        self.r_conv1 = FeaStConv(128, 64, 9)
        self.r_conv2 = FeaStConv(128, 64, 9)
        self.r_conv3 = FeaStConv(64, 32, 9)
        self.r_conv4 = FeaStConv(64, 32, 9)       
    def forward(self, data_r1, plot_pool=False):
        data_r1.x = torch.nn.functional.leaky_relu(self.l_conv1(data_r1.x, data_r1.edge_index), 0.2, inplace=True) # [N, 32]
        
        data_r2 = self.pooling1(data_r1)
        data_r2.x = torch.nn.functional.leaky_relu(self.l_conv2(data_r2.x, data_r2.edge_index), 0.2, inplace=True) # [N, 64]
        
        data_r3 = self.pooling2(data_r2)
        data_r3.x = torch.nn.functional.leaky_relu(self.l_conv3(data_r3.x, data_r3.edge_index), 0.2, inplace=True) # [N, 128]
        data_r3.x = torch.nn.functional.leaky_relu(self.l_conv4(data_r3.x, data_r3.edge_index), 0.2, inplace=True) # [N, 128]

        feat_r2_r = self.pooling2.unpooling(data_r3.x) # [N, 128]
        feat_r2_r = self.r_conv1(feat_r2_r, data_r2.edge_index) # [N, 64]

        data_r2.x = torch.cat((data_r2.x, feat_r2_r), 1) # [N, 128]
        data_r2.x = torch.nn.functional.leaky_relu(self.r_conv2(data_r2.x, data_r2.edge_index), 0.2, inplace=True) # [N, 64]

        feat_r1_r = self.pooling1.unpooling(data_r2.x) # [N, 64]
        feat_r1_r = self.r_conv3(feat_r1_r, data_r1.edge_index) # [N, 32]

        data_r1.x = torch.cat((data_r1.x, feat_r1_r), 1) # [N, 64]
        feat_r1_r = torch.nn.functional.leaky_relu(self.r_conv4(data_r1.x, data_r1.edge_index), 0.2, inplace=True) # [N, 32]
        return feat_r1_r
        
        
class PoolingLayer(torch.nn.Module):
    def __init__(self, in_channel, pool_type='max', pool_step=2, edge_weight_type=0, wei_param=2):
        super(PoolingLayer, self).__init__()
        assert (pool_type in ['max', 'mean'])
        self.pool_type = pool_type
        self.pool_step = pool_step
        self.edge_weight_type = edge_weight_type
        self.wei_param = wei_param

        if self.edge_weight_type in [4, 5]:
            self.lin = Linear(in_channel, in_channel)
        if self.edge_weight_type in [3, 4, 5]:
            # attention based edge weight for Graclus pooling
            self.att_l = Parameter(torch.Tensor(1, in_channel))
            self.att_r = Parameter(torch.Tensor(1, in_channel))
            init.xavier_uniform_(self.att_l.data, gain=1.414)
            init.xavier_uniform_(self.att_r.data, gain=1.414)

        self.unpooling_indices = None

    def forward(self, data, visual=False):
        val = data.edge_weight

        # yield data for Graclus pooling
        edge_weight = self._get_edge_weight(data)
        x, edge_index, pos = data.x, data.edge_index, data.pos
        edge_dual = data.edge_dual if hasattr(data, 'edge_dual') else None
        face = data.fv_indices if hasattr(data, 'fv_indices') else None

        # if visual and hasattr(data, 'fv_indices') and hasattr(data, 'name') and data.name[-1] == 'v':
        # if visual:
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), edge_weight.cpu().numpy())

        #     x_edge = x[edge_index]
        #     x_edge = ((x_edge[0] - x_edge[1])**2).sum(1)
        #     val = (x_edge/(-100)).exp()
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

        #     val = (x_edge/(-50)).exp()
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

        #     val = (x_edge/(-20)).exp()
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

        #     val = (x_edge/(-10)).exp()
        #     data_utils.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

        # pooling loop steps
        clusts = []
        for _ in range(self.pool_step):
            cluster = graclus(edge_index, edge_weight, x.shape[0])
            cluster, perm = consecutive_cluster(cluster)
            clusts.append(cluster)

            if self.pool_type == 'mean':
                x = scatter(x, cluster, dim=0, reduce='mean')
            elif self.pool_type == 'max':
                x = scatter(x, cluster, dim=0, reduce='max')
            edge_index, edge_weight = net_utils.pool_edge(cluster, edge_index, edge_weight)
            pos = None if pos is None else net_utils.pool_pos(cluster, pos)
            edge_dual = None if edge_dual is None else cluster[edge_dual]
            # if group all nodes to single one, i.e. there is no edge
            if edge_index.numel() == 0:
                break

            visual = False
            if visual and face is not None:
                face = None if face is None else net_utils.pool_face(cluster, face)
                import openmesh as om
                mesh = om.TriMesh(pos.cpu().numpy(), face.cpu().numpy())
                om.write_mesh(R"E:\SysFile\Desktop\pool_test.obj", mesh)
                pass



        # unpooling indices
        clust = clusts[-1]
        for c in clusts[-2::-1]:
            clust = clust[c]
        self.unpooling_indices = clust

        return torch_geometric.data.Data(x, edge_index, edge_dual=edge_dual, edge_weight=edge_weight, pos=pos, fv_indices=face)
    def _get_edge_weight(self, data):
        # remove self_loops
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        edge_index, edge_weight = remove_self_loops(data.edge_index, edge_weight)
        if edge_index.numel() == 0:
            return None
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        if self.edge_weight_type == -1:  # None, random
            edge_weight = None
        elif self.edge_weight_type == 0:  # 0. normal and spatial difference (like bilateral filtering)
            edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        elif self.edge_weight_type == 1:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            edge_weight = (feat_diff/(-self.wei_param)).exp()
        elif self.edge_weight_type == 2:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-self.wei_param)).exp()
            edge_weight = edge_weight * feat_diff
        elif self.edge_weight_type == 3:
            x = data.x
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            edge_weight = torch.nn.functional.sigmoid(alpha)
        elif self.edge_weight_type == 4:
            x = torch.nn.functional.leaky_relu(self.lin(data.x), 0.2, inplace=True)
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            edge_weight = F.sigmoid(alpha)
        elif self.edge_weight_type == 5:
            x = torch.nn.functional.leaky_relu(self.lin(data.x), 0.2, inplace=True)
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            wei = torch.nn.functional.sigmoid(alpha)
            edge_weight = (wei + data.edge_weight) / 2
        elif self.edge_weight_type == 6:
            edge_weight = (edge_weight-edge_weight.min()) / (edge_weight.max()-edge_weight.min()+1e-12)
        elif self.edge_weight_type == 7:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = -feat_diff
            edge_weight = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
        elif self.edge_weight_type == 8:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            edge_weight = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
        elif self.edge_weight_type == 9:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            feat_diff = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
            edge_weight = (edge_weight-edge_weight.min()) / (edge_weight.max()-edge_weight.min()+1e-12)
            edge_weight = edge_weight + feat_diff
        elif self.edge_weight_type == 10:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            edge_weight = edge_weight + feat_diff

        return edge_weight

    def unpooling(self, x):
        if self.unpooling_indices is not None:
            x = x[self.unpooling_indices]
        return x
