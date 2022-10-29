import torch.nn.functional as F
import numpy as np
import torch


a = torch.tensor([[1., -3., 5.], 
                  [1., -3., 5.]])

print(a.shape)

print(F.normalize(a, p=2, dim=0)) # tensor([ 1.0000, -0.0026])


# import albumentations as A  # torchvision을 대신할만한 라이브러리. image augmentation 등의 기능이 많고 빠르다

# mean1 = [90, 100, 100]
# std1 = [30, 32, 28]

# transform = A.Compose([
#     A.Normalize(mean=mean1, std=std1, max_pixel_value=1.0),
# ])
