import numpy as np
import torch
from torch import Tensor

t = torch.Tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

t2 = t[list(range(1,4)),:]

print(t2)
# bla = t.size()
# print(bla[0])
#
# print(t.reshape(1,3,2,2))