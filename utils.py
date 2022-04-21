import torch


def h_matmul(x):
    N = x.size()[-2]
    x_repeated_in_chunks = x.repeat_interleave(N, dim=-2)
    x_repeated_alternating = x.repeat(1, 1, N, 1)
    result = x_repeated_in_chunks * x_repeated_alternating
    return result


def similarity(x):
    a = h_matmul(x).sum(dim=-1, keepdim=True)
    b = torch.pow(x, 2).sum(dim=-1, keepdim=True)
    b = torch.pow(b, 0.5)
    p = a / (h_matmul(b))
    p = p.view(x.size()[0], x.size()[1], x.size()[2], -1)
    return p


"""
VIF and Correlation Function Module
"""

def mycorr(x):
    """
    tensor a value:
    tensor([[[[-2., -1.,  0.,  1.,  2.],
              [ 4.,  1.,  3.,  2.,  0.]]]])
    tensor a corr:
    tensor([[[[ 1.0000, -0.7000],
              [-0.7000,  1.0000]]]])
    numpy a_ value:
            [[-2 -1  0  1  2]
            [ 4  1  3  2  0]]
    numpy a_ corr:
            [[ 1.  -0.7]
            [-0.7  1. ]]
    """
    centered_h = x - x.mean(dim=-1, keepdim=True)
    covariance = h_matmul(centered_h).sum(dim=-1, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[-1] - 1)
    std_h = x.std(dim=-1, keepdim=True)
    p = bessel_corrected_covariance / (h_matmul(std_h))
    p = p.view(x.size()[0], x.size()[1], x.size()[2], -1)
    return p


def mysigmoid(x):
    """
    the modified sigmoid function which the function curve cross 0.5 at input x=1
    """
    return 1 / (1 + torch.exp((-x+1)))


def myvif(x):
    """
    Variance Inflation Factor (VIF)
    """
    corr = mycorr(x)
    pins = torch.linalg.pinv(corr)
    pins_d = torch.diagonal(pins, dim1=-2, dim2=-1)
    vif = mysigmoid(pins_d)
    return vif