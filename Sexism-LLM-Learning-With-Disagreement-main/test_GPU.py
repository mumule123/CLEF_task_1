import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
# 检查 PyTorch 是否支持 CUDA（即 GPU 版本）
print("CUDA 是否可用:", torch.cuda.is_available())

# 查看当前 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 查看 CUDA 版本（若可用）
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
else:
    print("PyTorch 是 CPU 版本")