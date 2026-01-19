# **LDM-MC-SFYOLO: Source-Free Domain Adaptation via Latent Diffusion & Uncertainty Estimation**

\<div align="center"\>

\<\!-- 将下面的链接替换为你实际的论文链接（如果有 arXiv） \--\>

**一种无需访问源域数据的鲁棒目标检测自适应框架，专为恶劣天气条件设计。**

\</div\>

## **📖 摘要 (Abstract)**

**LDM-MC-SFYOLO** 是一种新颖的无源域自适应 (Source-Free Domain Adaptation, SFDA) 框架，旨在解决在无法访问源域数据的情况下，将目标检测器适配到恶劣天气（如浓雾）场景中的难题。

本方法引入了两项核心创新：

1. **潜在扩散模型 (LDM) 增强**：替代传统的固定风格迁移方法，利用可控的文本驱动扩散模型动态生成多样化且逼真的天气效果，丰富训练样本。  
2. **不确定性感知伪标签 (Uncertainty-Aware Pseudo-Labeling)**：集成 **蒙特卡洛 (MC) Dropout** 技术来估计认知不确定性 (Epistemic Uncertainty)，有效过滤掉那些虽然置信度高但实际错误的伪标签，防止误差累积。

在 **Cityscapes** $\\to$ **Foggy Cityscapes** 基准测试中，我们的方法达到了 **55.7% mAP**，显著优于现有的基线方法。



## **📂 项目结构 (Project Structure)**

代码库结构经过精简，突出了核心贡献模块：

LDM-MC-SFYOLO/  
├── configs/               \# 数据集与超参数配置文件  
│   ├── source\_cityscapes.yaml  
│   └── target\_foggy\_cityscapes.yaml  
├── core/                  \# 核心算法实现 (本文贡献)  
│   ├── ldm\_augmenter.py   \# 基于 LDM 的数据增强模块  
│   └── uncertainty.py     \# MC Dropout 不确定性估计逻辑  
├── models/                \# 模型定义 (基于 YOLOv5 修改)  
│   ├── yolo.py  
│   └── sfyolo\_l.yaml      \# 主实验使用的 Large 模型配置  
├── scripts/               \# 一键复现脚本  
│   ├── download\_data.sh   \# 数据下载辅助脚本  
│   └── reproduce\_ablation.sh \# 复现消融实验  
├── tools/                 \# 可视化与辅助工具  
├── run\_adaptation.py      \# \[核心\] 域自适应训练入口脚本  
├── run\_pretrain.py        \# 源域预训练脚本  
├── evaluate.py            \# 模型评估脚本  
└── requirements.txt       \# 环境依赖

## **🛠️ 安装指南 (Installation)**

1. **克隆仓库**：  
   git clone \[https://github.com/YourUsername/LDM-MC-SFYOLO.git\](https://github.com/YourUsername/LDM-MC-SFYOLO.git)  
   cd LDM-MC-SFYOLO

2. **创建虚拟环境**：  
   conda create \-n sfyolo python=3.8 \-y  
   conda activate sfyolo

3. **安装依赖**：  
   \# 确保安装与 CUDA 11.8 匹配的 PyTorch 2.0  
   pip install torch torchvision \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)

   \# 安装核心依赖 (包含 diffusers, transformers 等)  
   pip install \-r requirements.txt

## **🚀 快速开始 (Quick Start)**

### **1\. 数据准备**

请下载 **Cityscapes** (源域) 和 **Foggy Cityscapes** (目标域) 数据集，并按如下结构放置：

datasets/  
├── CityScapes/  
└── CityScapesFoggy/

### **2\. 准备源域权重**

由于是无源域设置 (Source-Free)，我们需要一个在清晰图像上预训练好的模型权重。

* 请下载预训练权重 (yolov5l\_cityscapes.pt) 并放置在 source\_weights/ 文件夹中。  
* *\[下载链接请参考 source\_weights/README.md\]*

### **3\. 运行域自适应 (Run Adaptation)**

使用 LDM 增强和 MC Dropout 启动在目标域上的自适应训练：

python run\_adaptation.py \\  
    \--data configs/target\_foggy\_cityscapes.yaml \\  
    \--weights source\_weights/yolov5l\_cityscapes.pt \\  
    \--cfg models/sfyolo\_l.yaml \\  
    \--imgsz 640 \\  
    \--batch-size 16 \\  
    \--device 0,1 \\  
    \--ta\_method ldm \\  
    \--mc\_dropout \\  
    \--name experiment\_ldm\_mc

### **4\. 评估 (Evaluation)**

在测试集上评估训练好的模型：

python evaluate.py \\  
    \--data configs/target\_foggy\_cityscapes.yaml \\  
    \--weights runs/train/experiment\_ldm\_mc/weights/best\_teacher.pt \\  
    \--imgsz 960 \\  
    \--task test

## **📊 主要结果 (Main Results)**

在 **Cityscapes** $\\to$ **Foggy Cityscapes** 任务上的对比结果 (mAP@0.5)：

| 方法 (Method) | 骨干网络 (Backbone) | mAP (%) | 提升 (Improvement) |
| :---- | :---- | :---- | :---- |
| Source Only (仅源域) | YOLOv5l | 41.3 | \- |
| SF-YOLO (Baseline) | YOLOv5l | 51.2 | \+9.9% |
| **LDM-MC-SFYOLO (Ours)** | **YOLOv5l** | **55.7** | **\+14.4%** |

*注：以上结果基于上述硬件平台运行得出。*

## **🙏 致谢 (Acknowledgments)**

本项目基于以下优秀的开源项目构建，感谢原作者的贡献：

* **YOLOv5**: [Ultralytics](https://github.com/ultralytics/yolov5) \- 提供了强大的检测基线。  
* **SF-YOLO**: [Source-Free Domain Adaptation for YOLO](https://www.google.com/search?q=https://github.com/Extremedy/SF-YOLO) \- 提供了 SFDA 的基础框架。  
* **Diffusers**: [HuggingFace](https://github.com/huggingface/diffusers) \- 提供了 LDM 的实现支持。

## **📧 联系方式 (Contact)**

如果您对代码或论文有任何疑问，欢迎提交 Issue 或联系：your\_email@example.com。