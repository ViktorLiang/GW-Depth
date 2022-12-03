import torch
import torch.nn as nn
from src.ops import chamfer_2d

from src.ops.chamfer_2d import Chamfer2D


class ChamferLoss2D(nn.Module):
    def __init__(self, use_cuda=True, loss_weight=1.0, eps=1e-12, magrin=1.0):
        super(ChamferLoss2D, self).__init__()
        self.use_cuda = use_cuda
        self.loss_weight = loss_weight
        self.eps = eps
        self.margin = magrin

    def forward(self, point_set1, point_set2, point_set3):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        chamfer = Chamfer2D() if self.use_cuda else ChamferDistancePytorch()


        def onepair_set_loss(point_set_1, point_set_2):
            assert point_set_1.dim() == point_set_2.dim()
            assert point_set_1.shape[-1] == point_set_2.shape[-1]
            if point_set_1.dim() <= 3:
                if self.use_cuda:
                    dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
                    dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
                    dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
                    dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
                else:
                    dist = chamfer(point_set_1, point_set_2)
            else:
                point_dim = point_set_1.shape[-1]
                num_points_1, num_points_2 = point_set_1.shape[-2], point_set_2.shape[-2]
                point_set_1t = point_set_1.reshape((-1, num_points_1, point_dim))
                point_set_2t = point_set_2.reshape((-1, num_points_2, point_dim))
                if self.use_cuda:
                    dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
                    dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
                    dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
                    dist_t = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
                else:
                    dist_t = chamfer(point_set_1t, point_set_2t)
                dist_dim = point_set_1.shape[:-2]
                dist = dist_t.reshape(dist_dim)
            return self.margin - dist * self.loss_weight
        
        B = point_set1.shape[0]
        refer_points = point_set1.reshape(B, -1, 2)
        points2 = point_set2.reshape(B, -1, 2)
        points3 = point_set3.reshape(B, -1, 2)
        lss1 = onepair_set_loss(refer_points, points2)
        lss2 = onepair_set_loss(refer_points, points3)
        lss3 = onepair_set_loss(points2, points3)
        lss = (lss1 + lss2 + lss3) / 3 

        #gen_points = torch.cat([point_set2, point_set3], dim=1)
        #gen_points = gen_points.reshape(B, -1, 2)
        #t_lss_list = []
        #for b in range(B):
        #    b_lss = 0
        #    batch_points = gen_points[b]
        #    for pnt in batch_points:
        #        plss = onepair_set_loss(pnt[None, None], refer_points)
        #        b_lss = b_lss + plss
        #    b_lss = b_lss / len(batch_points)
        #    t_lss_list.append(b_lss)
        #lss = sum(t_lss_list) / B
        return lss

# Adapted from https://github.com/dfdazac/wassdistance
class ChamferDistancePytorch(nn.Module):
    r"""
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, reduction='none'):
        super(ChamferDistancePytorch, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if x.shape[0] == 0:
            return x.sum()
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # compute chamfer loss
        min_x2y, _ = C.min(-1)
        d1 = min_x2y.mean(-1)
        min_y2x, _ = C.min(-2)
        d2 = min_y2x.mean(-1)
        cost = (d1 + d2) / 2.0

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.norm(x_col - y_lin, 2, -1)
        return C

if __name__ == '__main__':
    #     def __init__(self, use_cuda=True, loss_weight=1.0, eps=1e-12):
    #     super(ChamferLoss2D, self).__init__()
    #     self.use_cuda = use_cuda
    #     self.loss_weight = loss_weight
    #     self.eps = eps

    # def forward(self, point_set_1, point_set_2):
    CL = ChamferLoss2D(use_cuda=True, loss_weight=1.0)
    ps1 = torch.randn(2, 10, 2).cuda()
    ps2 = torch.randn(2, 10, 2).cuda()
    a = CL(ps1, ps2)
    print(a)
