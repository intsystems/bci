import torch
import tensorly as tl

tl.set_backend('pytorch')

def hosvd(
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