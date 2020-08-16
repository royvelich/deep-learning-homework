import torch
import torch.utils.data
import numpy

t1 = torch.tensor([4]);
# t1.unsqueeze(0)
print(t1.shape)

t2 = torch.tensor([8]);
# t2.unsqueeze(0)
print(t2.shape)
# t2 = torch.IntTensor(2);

states_tensor = torch.cat((t1, t2), 0)

h = 5
