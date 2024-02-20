import torch
import math
try:
    from pytorch3d.ops.points_alignment import iterative_closest_point as icp
except ImportError:
    icp = None

def loss_v(vp, v, dis='L2', apply_icp=False):
    if apply_icp:
        icp_rst = icp(vp.unsqueeze(0), v.unsqueeze(0))
        vp = icp_rst.Xt.squeeze()

    if dis == 'L1':
        loss = (vp-v).abs().sum(1).mean()
    elif dis == 'L2':
        loss = (vp-v).pow(2).sum(1).mean()
    return loss


def loss_n(np, n, norm='L1', fc_p=None, fc=None):
    if norm == 'L1':
        loss = (np-n).abs().sum(1).mean()
    elif norm == 'L2':
        loss = (np-n).pow(2).sum(1).mean()

    return loss


def dual_loss(loss_v, loss_n, v_scale=1, n_scale=1, alpha=None):
    if alpha is None:
        return loss_v*v_scale + loss_n*n_scale
    else:
        return alpha*loss_v*v_scale + (1-alpha)*loss_n*n_scale
    
def discriminator_loss(real_vertex_logits, real_facet_logits, fake_vertex_logits, fake_facet_logits,device):

    # vertex loss
    target_real_vertex = torch.FloatTensor(real_vertex_logits.size()).uniform_(0.9, 1.0).to(device)
    target_fake_vertex = torch.FloatTensor(real_vertex_logits.size()).uniform_(0.0, 0.1).to(device)
    real_vertex_loss = (real_vertex_logits - target_real_vertex).pow(2).sum(1).mean()
    fake_vertex_loss = (fake_vertex_logits - target_fake_vertex).pow(2).sum(1).mean()

    # facet loss
    target_real_facet = torch.FloatTensor(real_facet_logits.size()).uniform_(0.9, 1.0).to(device)
    target_fake_facet = torch.FloatTensor(real_facet_logits.size()).uniform_(0.0, 0.1).to(device)
    real_facet_loss = (real_facet_logits - target_real_facet).pow(2).sum(1).mean()
    fake_facet_loss = (fake_facet_logits - target_fake_facet).pow(2).sum(1).mean()

    total_loss = real_vertex_loss+fake_vertex_loss+real_facet_loss+fake_facet_loss
    
    return total_loss*0.25

def generator_loss(fake_vertex_logits, fake_facet_logits, device):

    # vertex loss
    target_real_vertex = torch.FloatTensor(fake_vertex_logits.size()).uniform_(0.9, 1.0).to(device)
    vertex_loss = (fake_vertex_logits - target_real_vertex).pow(2).sum(1).mean()
    
    # facet loss
    target_real_facet = torch.FloatTensor(fake_facet_logits.size()).uniform_(0.9, 1.0).to(device)
    facet_loss = (fake_facet_logits - target_real_facet).pow(2).sum(1).mean()
    
    total_loss = vertex_loss+facet_loss

    return total_loss*0.25

def error_v(vp, v):
    """
    Euclidean distance
    """
    error = (vp-v).pow(2).sum(1).pow(0.5)
    return error.mean()


def error_n(np, n):
    """
    Intersection angular
    """
    error = (np-n).pow(2).sum(1)
    val = torch.clamp(1-error/2, min=-1, max=1)
    return (torch.acos(val) * 180 / math.pi).mean()