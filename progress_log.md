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