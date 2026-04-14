🏆 CVPR 2026 AV2 Challenge - Progress Log
Date: 2026-04-12
Phase: Environment Setup & Data Mining
Status: 🚀 Engine Started

🛠️ Today's Major Achievements
Framework Deployment (WSL2/Linux):

Successfully compiled OpenPCDet v0.6.0 from source in a WSL2 environment.

Resolved CUDA/C++ extension compatibility issues for NVIDIA RTX 5070 (12GB).

Achieved a clean build of sparse convolution kernels (SpConv), unlocking full GPU acceleration for 3D detection.

Strategic Data Mining (Data-Centric AI):

Designed and implemented a high-performance Scenario Miner based on TTC (Time-to-Collision) physics.

Targeting high-risk scenarios: VRU_CRITICAL (Pedestrian near-misses) and VEHICLE_RISK (High-speed car-following conflicts).

Optimized data pipeline for the massive AV2 Sensor Dataset (199,988 scenarios) using fastparquet and tqdm for real-time monitoring.

Model Configuration:

Selected VoxelNeXt as the primary backbone (CBGS enabled) to maximize 3D spatial awareness.

Optimized Training YAML: Set BATCH_SIZE_PER_GPU: 4 and planned for 30 Epochs to overfit on hard cases.

🧠 Reflections & Insights (今日反思)
Engineering over Brute Force: 面对 20 万个场景（59GB），盲目全量训练是低效的。通过物理规则（TTC）先进行“数据蒸馏”，能让模型在有限的算力下更快学习长尾分布（Corner Cases）。

System Bottlenecks: WSL2 跨文件系统（/mnt/c/）读取大量小文件时 I/O 损耗显著。未来如果数据量继续翻倍，需考虑将数据集直接迁入 Linux Native Drive (ext4)。

The Power of Real-time Feedback: 增加了流式保存（Streaming Save）和实时进度条，不仅是心理安慰，更是为了在长时间任务中防止数据丢失并允许“边挖边练”。
📅 2026-04-13 Progress Log
Phase:
 CUDA Compilation & System Limit Pushing
Status: 🚧 Hitting the Hardware Wall
🚀 Today's Milestones
• Hardware Limit Bypass: Encountered severe OOM (Out-of-Memory) Killed errors during the compilation of complex CUDA operators (pointnet2_stack, iou3d_nms) due to the 8GB physical RAM constraint.
• System-Level Optimization: Successfully bypassed the hardware bottleneck by manually configuring a 16GB Swap File via Linux native commands, allowing the SSD to act as an extended memory pool for the C++ compiler.
• Single-Thread Compilation: Forced MAX_JOBS=1 to ensure stable compilation, successfully building the OpenPCDet core library without system crashes.
🧠 Reflections & Insights (今日反思)
• Hardware is the Ceiling, but Software is the Ladder: 8GB 内存对于本地编译 3D 检测算子确实是极限挑战，但通过合理的 OS 级资源调度（Swap 挂载 + 单核限流），完全可以做到“以时间换空间”。
• WSL2 vs Native Linux: Cross-OS environment variables and Python pathing (e.g., /mnt/c/ vs /home/) can cause severe referencing errors. Strict environment management (PYTHONPATH locking) is crucial.
￼
📅 2026-04-14 Progress Log
Phase:
 DataLoader Debugging & Final Ignition
Status: 🏆 Training Commenced!
🚀 Today's Milestones
• OpenPCDet 0.6.0 Code Patching: Resolved a critical ImportError (circular dependency) caused by collate_batch in the new OpenPCDet version by directly injecting the data collation logic into argo2_dataset.py.
• The Ultimate Breakthrough (Root Cause Analysis): Solved the persistent FileNotFoundError in the DataLoader. Discovered a massive discrepancy in dataset modalities: The pipeline was mistakenly feeding the AV2 Map/Log Dataset into the model instead of the AV2 Sensor Dataset.
• Data Pipeline Rectification: * Transitioned data source to authentic LiDAR .bin point cloud files.
• Rewrote the create_argo2_infos.py script to strictly validate and index .bin files, successfully loading 53,455 training samples.
• Engine Ignition: Successfully initialized the 7.88 Million parameter VoxelNeXt model on the RTX 5070. Epoch 0 started rendering losses successfully.
🧠 Reflections & Insights (今日反思)
• Data Validity > Code Logic (数据源头决定一切): 我们花了大量时间在代码层面排查路径、软链接和内存问题，结果根源在于“喂错了数据类型”（Map vs Sensor）。在深度学习中，如果数据本身是错的，再精妙的 Debug 都是徒劳。遇到底层读写报错，永远先去 Check 原始数据格式！
• The Value of Persistence: 连续几天的通宵排错极度消耗精力，但通过阅读源码、查阅官方 Dataset 文档并重写底层脚本，我对整个 3D 检测数据管道的理解达到了前所未有的深度。CVPR 2026, here we go!
