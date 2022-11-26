import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch

image_embeddings = torch.tensor([[0,1.,1.,1.],
                             [3.,5.,6.,1.]])
text_embeddings = torch.tensor([[2.,3.,4.,5.],
                            [4.,5.,6.,7.]])

img_s = image_embeddings @ image_embeddings.T
text_s = text_embeddings @ text_embeddings.T

logits = text_embeddings @ image_embeddings.T
logits_s = torch.softmax(logits, dim=-1)

target = (img_s + text_s) / 2
target_s = torch.softmax((img_s + text_s) / 2, dim=-1)

print(img_s)
# print(target)
# print(target_s)
# print(logits)
# print(logits_s)


# tensor([[28.5000, 47.0000],
#         [47.0000, 98.5000]])
# tensor([[9.2374e-09, 1.0000e+00],
#         [4.3036e-23, 1.0000e+00]])

# tensor([[12., 50.],
#         [18., 80.]])
# tensor([[3.1391e-17, 1.0000e+00],
#         [1.1851e-27, 1.0000e+00]])
