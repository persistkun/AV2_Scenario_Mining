
# 🚀 AV2 场景挖掘实录：在 RTX 5070 (Blackwell) 上彻底驯服 OpenPCDet

本仓库记录了在 **Argoverse 2 (AV2)** 数据集上，使用 **OpenPCDet** 框架训练 VoxelNeXt 模型时遇到的极致踩坑与环境配置全过程。

特别针对 **RTX 50 系列显卡 (Blackwell 架构, sm_120)** 和 **WSL2 小内存环境** 提供了目前全网领先的解决方案。

---

## 💻 硬件与环境底座 (Hardware & Environment)
* **OS:** Windows 11 + WSL2 (Ubuntu 22.04)
* **GPU:** NVIDIA GeForce RTX 5070 Laptop (最新 Blackwell 架构, 算力 **sm_120**)
* **RAM:** 8GB 物理内存 (通过虚拟内存扩展)
* **Framework:** OpenPCDet v0.6.0
* **Dataset:** Argoverse 2 Sensor Dataset (包含 .bin 与 .feather 格式)

---

## 🚧 核心挑战与终极解决方案 (The Hurdles & Solutions)

### 1. 💀 路径陷阱与数据读取 (Pathing & Data Format)
**问题描述**：OpenPCDet 的配置文件嵌套极深，生成 `.pkl` 索引后，DataLoader 频繁出现路径拼接错误（例如多出 `/data/av2/data/sensor/`），导致 `FileNotFoundError`。此外，AV2 原始数据包含 `.feather` 格式，原生代码无法直接读取。
**解决方案**：
摒弃 YAML 相对路径逻辑，在 `argo2_dataset.py` 中采用“绝对路径焊死”策略，并集成 Pandas 读取功能。
```python
# 核心修复逻辑 (pcdet/datasets/argo2/argo2_dataset.py)
# 1. 强制重定向 info_path 到物理绝对路径
self.info_path = '/mnt/c/AV2_Scenario_Mining/OpenPCDet/data/infos/argo2_infos_train.pkl'

# 2. 截断异常拼接，强制纠偏雷达数据目录
# 防止出现 .../data/av2/data/sensor/... 的套娃路径
if "sensor/train" in raw_path:
    lidar_path = "/mnt/c/AV2_Scenario_Mining/OpenPCDet/data/sensor/train" + raw_path.split("sensor/train")[-1]

# 3. 兼容 Feather 读取
if lidar_path.endswith(".feather"):
    points = pd.read_feather(lidar_path).values.astype(np.float32)
```

### 2. 💥 物理内存榨干 (OOM killed during Compilation)
**问题描述**：执行 `python setup.py develop` 编译 CUDA 算子时，由于 8GB 物理内存瞬间被多线程挤爆，进程会被 Linux 内核直接 `Killed`。
**解决方案**：
采用“虚拟内存挂载 + 单核强制锁死”战术，以时间换空间。
```bash
# 1. 向硬盘借 16GB 空间作为虚拟内存 (Swap)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. 强制单线程编译 (防止多核并发导致内存溢出)
export MAX_JOBS=1
python setup.py develop
```

### 🛑 3. 终极 Boss：RTX 5070 架构不兼容 (`sm_120`)
**问题描述**：环境配置完成后，启动训练即报 `RuntimeError: CUDA error: no kernel image is available for execution on the device`。
**原因剖析**：RTX 50 系列是全新的 **Blackwell** 架构，算力为 **sm_120**。目前所有正式版 PyTorch (<=2.5) 均不支持此架构指令集。
**解决方案**：
全线升级至 PyTorch Nightly 预览版环境，并手动指定算力进行重构。
```bash
# 1. 卸载旧版 Torch 与 spconv
pip uninstall -y torch torchvision torchaudio spconv-cu118

# 2. 安装支持 sm_120 的 PyTorch Nightly (必须包含 CUDA 12.6/12.8 支持)
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

# 3. 安装适配新驱动的 spconv-cu120 
pip install spconv-cu120

# 4. 显式声明算力列表并重新编译算子
rm -rf build/
python setup.py clean
export TORCH_CUDA_ARCH_LIST='12.0'
MAX_JOBS=1 python setup.py develop
```

---

## 🏆 训练实测表现 (Training Results)
在资源有限的 8G 内存笔记本环境下，采用如下安全参数成功启动训练：
```bash
cd tools
export PYTHONPATH=$PYTHONPATH:..
# workers=2 兼顾速度与内存稳定性，batch_size=2 适配 8G 显存
python train.py --cfg_file cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml --batch_size 2 --workers 2
```
**运行状态监控**：
* **计算效率**：RTX 5070 表现强劲，`Forward time` 约 0.4s，硬件利用率极高。
* **收敛情况**：训练起始 Loss `1310`，在 150 个 Iteration 后迅速收敛至 `22.x`。
* **脏数据清洗**：通过 `np.nan_to_num` 处理，有效防止了 AV2 标注中的空值导致训练崩溃。

---

## 💡 给开发者的避坑总结 (Takeaways)
1. **拥抱 Nightly 版**：如果你是 RTX 50 系列的第一批用户，正式版 PyTorch 大概率会报 `no kernel image`，直接上 Nightly 版是唯一的出路。
2. **绝对路径是救星**：在 WSL2 与 Windows 跨系统文件交互时，Hardcode（硬编码）绝对路径能节省 90% 的路径调试时间。
3. **内存不足 Swap 凑**：8G RAM 的机器，如果不挂载 16G 以上的 Swap 且不限制 `MAX_JOBS=1`，编译 3D 算子几乎 100% 会失败。

---
*记录人：一名在 AV2 赛道上死磕到底，最终成功驯服 RTX 5070 的炼丹师。*
```

---

**兄弟，这就是你这几天的战斗成果！** 去把它存好，然后关机睡觉。明天开机重新“点火”的时候，你就是一个拥有完美环境的大佬了！祝你早日练出冠军模型，晚安！🌙🚀
