from torch.nn import L1Loss, MSELoss
import torch
import collections


l1_loss = L1Loss()
l2_loss = MSELoss()

torch_1 = torch.randn((3,3,512,512))
torch_2 = torch.randn((3,3,512,512))


torch_3 = torch.nn.init.normal_(torch.empty(torch_1.shape))


a_dict = collections.OrderedDict()
a_dict['m'] = 10
a_dict['u'] = 15
a_dict['c'] = 5
a_dict['h'] = 25
a_dict['a'] = 55
a_dict['s'] = 30
import copy
b_dict = copy.deepcopy(a_dict)
b_dict['s'] = 50
print(a_dict)
print(b_dict)