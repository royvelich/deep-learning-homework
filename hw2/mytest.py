import torch

x = [1,2,3,4,5,6,7]

y = x[0:2]

print(y)

# x = torch.rand(3).expand(2,-1)
# print(x)

# mylist = [(1,2), (3,4), (5,6)]
#
# for idx, (i,j) in enumerate(mylist):
#     print(idx)
#     print(i)
#     print(j)

# mylist = [1,2,3]
# for item in mylist:
#     item = 5
#
# print(mylist)


# x = torch.Tensor([1,2,3]).unsqueeze(1)
# y = x.expand(-1,3)
# print(y)

# x = torch.Tensor([[1, 2, 3], [-4, 5, -1], [-1, 8, -1]])
# y = x.argmax()
# print(y)

#
# w = x.exp().sum(dim=0).log()
#
# print(w)
#
# y = torch.Tensor([1,2,2]).to(dtype=torch.long)
#
# z = x[range(3), y]
#
# print(z)

# zero = torch.zeros_like(x)
# y = x.max(zero)


# x = torch.Tensor([[1, 2], [3,4]])
#
# w = 1 / (1 + (-x).exp())


# print(w)