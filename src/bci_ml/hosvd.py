import torch
import tensorly as tl

tl.set_backend('pytorch')

def hosvd_t(
    t,
    new_shape
):
    """
    Higher Order SVD Realisation
    Params
    ----------
    t : torch.Tensor,
        dimensional tensor
    new_shape : Tuple,
        new shape for tensor
    Returns
    -------
    torch.Tensor
        core tensor
    List[torch.Tensor]
        singular vectors
    List[torch.Tensor]
        singular values
    """
    sin_vecs = []
    sin_vals = []

    for i in range(len(new_shape)):

        unfold = t
        s_vec, s_val, _ = torch.linalg.svd(tl.unfold(unfold, i))
        t = tl.tenalg.mode_dot(t, s_vec.t(), i)
        sin_vecs.append(s_vec)
        sin_vals.append(s_val)

    return t, sin_vecs, sin_vals

"""
Example:

a = torch.randn(5, 3)
print(a)

tensor([[ 0.6416,  1.2616, -0.5460],
        [ 0.4539,  1.9186, -0.1070],
        [-0.2948, -0.3432,  0.6857],
        [ 1.8485, -0.8002,  0.5453],
        [ 1.2835,  0.7288, -0.2886]])
        
hosvd_t(a, (2,3))

(tensor([[ 2.7903e+00, -2.2573e-06,  1.7060e-07],
         [-2.6474e-06, -2.2600e+00, -6.8816e-08],
         [ 3.3963e-07,  1.2208e-07, -7.7335e-01],
         [-1.5910e-07,  4.3793e-08,  1.0520e-07],
         [-1.3928e-07,  6.9593e-08, -1.2432e-08]]),
 [tensor([[-0.5365,  0.0684,  0.2479, -0.5949, -0.5405],
          [-0.6638,  0.2328, -0.5654,  0.4286, -0.0428],
          [ 0.2103, -0.0246, -0.7401, -0.6238,  0.1354],
          [-0.0540, -0.9190, -0.1733,  0.1556, -0.3134],
          [-0.4738, -0.3097,  0.2028, -0.2215,  0.7678]]),
  tensor([[-0.5073,  0.8582, -0.0784],
          [-0.8331, -0.4651,  0.2995],
          [ 0.2205,  0.2172,  0.9509]])],
 [tensor([2.7903, 2.2600, 0.7734]), tensor([2.7903, 2.2600, 0.7734])])
 """
