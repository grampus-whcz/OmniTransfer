import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import torch
print(torch.version.cuda)  # 例如：11.8
print(torch.cuda.is_available())  # 应为 True（如果你有 GPU）

from torch_geometric.nn import GCNConv  # 应该能正常导入