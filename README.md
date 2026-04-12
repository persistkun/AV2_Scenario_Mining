# CVPR 2026 - Argoverse 2 Scenario Mining Project

## 📅 2026-04-12 Progress Log

### 🚀 Today's Milestones
- **Infrastructure**: Successfully migrated the development environment to **WSL2 (Ubuntu)**.
- **Hardware Optimization**: Compiled **OpenPCDet** with SpConv 2.x, fully utilizing the **NVIDIA RTX 5070** for 3D detection tasks.
- **Data Mining**: Developed a Physics-based Scenario Miner. 
  - **Algorithm**: Time-to-Collision (TTC) & VRU Criticality analysis.
  - **Throughput**: Processing **199,988** scenarios from the AV2 Sensor Dataset.
  - **Yield**: Already captured **20,000+** high-difficulty scenarios for targeted training.

### 🧠 Reflections
- **Data-Centric Strategy**: Training on 20k "hard cases" is far more efficient than brute-force training on 200k random frames.
- **System Tuning**: Identified I/O bottlenecks during cross-system file access; implemented streaming writes to prevent data loss.

### 🎯 Next Steps
- Start training **VoxelNeXt** on the extracted hard-case subset.
- Monitor Loss curves and evaluate the mAP improvement on long-tail categories.