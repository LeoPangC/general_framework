import torch
import torch.nn.functional as F

mask = torch.tensor([[True, False, True, False],
                     [False, True, False, True],
                     [True, False, True, False]])
mask = mask[:, None, :] * mask[:, :, None]

print(mask.shape)