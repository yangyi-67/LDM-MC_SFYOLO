
🌟 核心亮点
针对无源域自适应（SFDA）中目标域样本多样性不足和**伪标签过自信错误（OEs）**两大难题，我们提出了 LDM-MC-SFYOLO 协同框架 ：


Dynamic LDM Augmentor: 首次将预训练的潜在扩散模型（LDM）引入 SFDA 领域 。通过基于目标域统计特性的动态提示词库，在无需额外训练的情况下合成精细化、可控的增强样本 。





Uncertainty-Aware Mean Teacher: 设计了基于蒙特卡洛（MC）Dropout 的不确定性量化机制，综合考虑了分类和定位的方差 。



Dual-Threshold Filtering: 采用双阈值过滤策略，通过“动态生成-不确定性验证”闭环，有效剔除过自信的噪声标签，仅保留经过良好校准的预测（WCPs） 。

🛠 架构概览图 1: LDM-MC-SFYOLO 协同训练框架图 8🚀 快速开始1. 环境准备所有实验均在 Ubuntu 20.04、PyTorch 2.0 和 CUDA 11.8 环境下完成 9。Bash# 克隆仓库
git clone https://github.com/yangyi-67/LDM-MC_SFYOLO.git
cd LDM-MC_SFYOLO

# 安装依赖
pip install -r requirements.txt
2. 数据集配置本项目在以下三个标准 SFDA 场景上进行了验证 10101010：C2F: Cityscapes $\rightarrow$ Foggy Cityscapes (模拟恶劣天气) 11K2C: KITTI $\rightarrow$ Cityscapes (跨相机配置) 12S2C: Sim10k $\rightarrow$ Cityscapes (合成到现实) 133. 训练流程使用训练好的源域模型启动无源自适应训练：Bashpython train_sfda.py --cfg models/yolov5l_sfda.yaml --data data/c2f.yaml --weights source_only_v5l.pt --img 640
📊 实验结果LDM-MC-SFYOLO 在各大基准测试中均达到了 SOTA 性能 141414141414141414：场景方法基准模型性能 (mAP/AP)C2FSF-YOLO (Baseline)YOLOv5l51.2% 15C2FLDM-MC-SFYOLO (Ours)YOLOv5l55.7% 16K2CLDM-MC-SFYOLO (Ours)YOLOv5l65.9% 17S2CLDM-MC-SFYOLO (Ours)YOLOv5l72.8% 18注：在 K2C 场景下，我们的无源方法甚至超越了有源 UDA 方法 SSDA-YOLO (60.5% AP) 19。🖼️ 定性分析对比(a) Source-only: 漏检严重；(b) SF-YOLO: 存在误检；
