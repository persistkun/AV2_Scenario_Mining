import torch
import av2
import os
import numpy as np
import pandas as pd

print("--- 🎉 恭喜！新环境配置成功 ---")
print(f"当前项目路径: {os.getcwd()}")
print(f"显卡状态: {'✅ RTX 5070 已点亮' if torch.cuda.is_available() else '❌ 显卡未正常工作'}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"AV2工具包版本: 0.3.6")
print(f"NumPy版本: {np.__version__}")
print(f"Pandas版本: {pd.__version__}")
print("\n--- ✅ 环境验证通过，可以开始跑Baseline了！---")