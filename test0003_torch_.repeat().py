import numpy as np
import torch

a = torch.arange(0, 8)
print(a)

a = a.unsqueeze(0)
print(a)

a= a.repeat(3, 2)
print(a)

a = 2
b = 2
print(a&b)