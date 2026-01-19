# **LDM-MC-SFYOLO: A Synergistic Framework for Source-Free Domain Adaptive Object Detection via Latent Diffusion and Uncertainty Estimation**


## **📖 Overview**

LDM-MC-SFYOLO is a novel Source-Free Domain Adaptation (SFDA) framework designed to adapt one-stage object detectors to adverse weather conditions (e.g., dense fog) without accessing original source data.



The framework introduces two synergistic innovations:


Dynamic LDM Augmentation: Replaces static style transfer by leveraging a frozen pre-trained Latent Diffusion Model (LDM). It uses a dynamic prompt library derived from target domain statistics (via Dark Channel Prior) to synthesize fine-grained and diverse samples in a zero-shot manner.

Uncertainty-Aware Pseudo-Labeling: Integrates Monte Carlo (MC) Dropout to quantify epistemic uncertainty by fusing classification and localization variances into a composite metric. A dual-threshold filtering strategy is then employed to effectively eliminate Overconfident Errors (OEs) and retain Well-Calibrated Predictions (WCPs).

This architecture establishes a mutually reinforcing closed loop of "Dynamic Generation and Uncertainty Verification".


## 📂 Project Structure

The repository is organized to highlight the core contributions:

```text
LDM-MC-SFYOLO/
├── TargetAugment/         # Target Augmentation Module (TAM)
│   ├── enhance_ldm.py    # [Core] LDM-based dynamic style generation
│   └── enhance_vgg16.py  # VGG-based style transfer (Baseline)
├── utils/                 # Utility functions and core logic
│   ├── mc_dropout.py     # [Core] MC Dropout & uncertainty estimation
│   └── loss.py           # Modified YOLOv5 loss function for SFDA
├── models/                # Architecture definitions
│   ├── yolo.py           # YOLOv5 model construction
│   └── yolov5l.yaml      # Configuration for the Large backbone
├── run_adaptation.py     # [Main] Entry point for SFDA training
├── run_pretrain.py       # Script for source-domain pre-training
└── requirements.txt      # Environment dependencies
```

## **🛠️ Installation**

1. **Clone the Repository:**：  
   git clone https://github.com/yangyi-67/LDM-MC_SFYOLO.git
   cd LDM-MC_SFYOLO

2. **Create Environment**：  
   conda create \-n sfyolo python=3.8 \-y  
   conda activate sfyolo

3. **Install Dependencies**：  
   \# We recommend using PyTorch 2.0+ and CUDA 11.8+ for LDM inference.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

## **🚀 Quick Start**

### **1\.Data Preparation**

Organize your datasets (Cityscapes, Foggy Cityscapes, KITTI, Sim10k) as follows:：
```text
datasets/
├── Cityscapes/
├── Foggy_Cityscapes/
├── KITTI/
└── Sim10k/
```

### **2\. Source Pre-training**
Train a baseline model on the source domain:
python run_pretrain.py --weights yolov5s.pt --data configs/source_cityscapes.yaml --imgsz 960 --epochs 100


### **3\. Source-Free Domain Adaptation**
Launch the adaptation process using the synergistic framework：

python run_adaptation.py \
    --weights path/to/source_model.pt \
    --data configs/target_foggy_cityscapes.yaml \
    --ta_method ldm \
    --ldm_strength 0.35 \
    --mc_dropout --mc_T 10 \
    --conf_thres 0.4 --iou_thres 0.3 \
    --teacher_alpha 0.999 \
    --SSM_alpha 0.0 \
    --batch-size 16 \
    --device 0


## **🙏 Acknowledgments**
We thank the authors of YOLOv5, SF-YOLO, and HuggingFace Diffusers for their excellent open-source contributions!

## **📧 Contact**
For any questions, please open an issue or contact: s202420211022@stu.tyust.edu.cn.
