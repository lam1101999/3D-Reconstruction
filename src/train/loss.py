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