import numpy as np
import torch
from torch import Tensor


# mat = torch.randn(64, 512)

# mat2 = torch.randn(2, 3, 4)
#
# mat2 = torch.randint(low=0, high=10, size=(1, 12)).to(dtype=float)
#
# # bla = mat2.size(0)
#
# dims = mat2.size()
# # dims = dims[:len(dims)-1]
#
# first_dims_sizes = dims[:len(dims)-1]
# last_dim_index = len(dims) - 1
#
# y = torch.ones(*first_dims_sizes).unsqueeze(last_dim_index).to(dtype=float)
#
# z = torch.cat((y,mat2), dim=last_dim_index).to(dtype=float)








# mat2 = torch.randn(2, 3, 4)
mat2 = torch.randint(low=0, high=10, size=(1, 12))
dims = mat2.size()
first_dims = dims[:len(dims)-1]
first_dims = [*first_dims, 1]
last_dim_index = len(dims) - 1
ones = torch.ones(*first_dims).to(dtype=mat2.dtype)
z = torch.cat((ones,mat2), dim=last_dim_index)
bla = 5