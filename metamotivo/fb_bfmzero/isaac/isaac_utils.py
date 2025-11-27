import torch
import isaaclab.utils.math as math_utils

def quat_to_tan_norm_wxyz2(q):
    lb_mat = math_utils.matrix_from_quat(q)
    lb_mat = lb_mat[..., [0, 2]]
    lb_mat2 = torch.transpose(lb_mat, 2, 1)
    return lb_mat2.reshape(lb_mat2.shape[0], -1)

@torch.jit.script
def quat_to_tan_norm_wxyz(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 1:])
    ref_tan[..., 0] = 1
    tan =math_utils.quat_apply(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 1:])
    ref_norm[..., -1] = 1
    norm = math_utils.quat_apply(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

