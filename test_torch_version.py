import torch

print(torch.version.cuda)  # 查看CUDA版本
print(torch.cuda.is_available())  # 检查GPU是否可用