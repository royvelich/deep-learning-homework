import numpy as np
import torch
from torch import Tensor


left = torch.range(0, 3).unsqueeze(0)
right = torch.range(2, 5).unsqueeze(0)

bibi = torch.cat((left,right), dim=0)

bibi = torch.cat((left,right),dim=1)

mat = torch.Tensor([[-1,2,3],[5,6,7],[7,-8,9]])

mat[[0,1,2],[1,1,1]] += 3

dodo = torch.Tensor([0,2,1]).unsqueeze(1)

indices = torch.Tensor([[0],[2],[1]]).to(dtype=int)
values = torch.gather(mat,1,indices)
mat2 = values.expand(3,6)

mat2_sub = mat2 + 3
max = torch.max(mat2_sub, torch.zeros_like(mat2_sub))
yoyo = torch.sum(max,dim=1)
bla = 5