# [TKDE 2026] Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Model

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/LivXue/short-video-influence?style=social)](https://github.com/LivXue/short-video-influence/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-TKDE%202026-red.svg)](https://ieeexplore.ieee.org/abstract/document/11434967/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)

**Official implementation of the TKDE 2026 paper**
*Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Model*

</div>

> ⚠️ **Work in Progress**
>
> This repository is currently under active construction. Some features may be incomplete, and you may encounter issues when running the code directly. We are working to improve the documentation and code stability, and a fully usable version will be released soon.

---

## 🔥 Overview
This work introduces two key contributions:
1. **XS-Video**: A new large-scale real-world short video dataset with complete user-video interaction graphs
2. **Novel Multimodal Graph LLM**: An end-to-end framework that integrates Qwen2-VL multimodal large language model with RGCN graph neural network to capture both content semantics and social propagation context for accurate short video influence prediction.

Our model achieves state-of-the-art performance on short video influence rating tasks by unifying visual content analysis, text understanding, and social graph modeling.

---

## ✨ Key Features
- 🎥 **Multimodal Fusion**: Seamlessly combines video content, text metadata, and heterogeneous social graph features
- 📊 **XS-Video Dataset**: Large-scale real-world dataset with 100k+ short videos and their complete propagation interaction graphs
- 🧠 **Hybrid Architecture**: Custom modified Qwen2-VL with extended graph feature input support + RGCN graph encoder
- ⚡ **Efficient Training**: Distributed training with DeepSpeed ZeRO Stage 2, BF16 mixed precision, and Flash Attention 2 optimization
- 📈 **Strong Performance**: Outperforms existing baseline methods on both influence classification and regression tasks

---

## 📦 Installation

### 📂 Dataset Preparation
The XS-Video dataset raw files are provided in `dataset/`. You can generate the processed graph-structured dataset with:
```bash
python group_dataset.py
```
> ⚠️ This preprocessing step may take several hours to complete. For convenience, you can directly download our preprocessed graph dataset from [Google Drive](https://drive.google.com/file/d/1PqVVilGfgkvgVYrrKjXep_QQMbnm0ndJ/view?usp=sharing).

### 🔧 Environment Setup
```bash
# Clone the repository
git clone https://github.com/LivXue/short-video-influence
cd short-video-influence

# Install all dependencies
pip install -r requirements.txt

# Install custom transformers with NetQwen2VL graph support
cd transformers
pip install -e .
cd ..
```
> 💡 Requires CUDA 12.1+ and GPU with minimum 80GB VRAM for 4-GPU distributed training.

---

## 🚀 Usage

### 🏋️ Training
The model training follows three progressive stages:

#### Stage 1: Heterogeneous Graph Pretraining
Train the RGCN graph encoder on the XS-Video interaction graph to learn structural node representations:
```bash
python rgcn/train_RGCN_node_classification.py
```

#### Stage 2: Supervised Language Fine-tuning
Train the graph feature projector to align graph embeddings with the Qwen2-VL multimodal space:
```bash
deepspeed --num_gpus 4 train_single.py --deepspeed --deepspeed_config deepspeed_config_single.json
```

#### Stage 3: Task-oriented End-to-end Fine-tuning
Fine-tune the entire unified model for the final influence prediction task:
```bash
deepspeed --num_gpus 4 train_single_stage2.py --deepspeed --deepspeed_config deepspeed_config_single.json
```

> 💾 Training checkpoints are automatically saved to the `NetQwen2-VL-7B-final/` directory.

### 🧪 Inference & Evaluation
Run evaluation on the held-out test dataset to get performance metrics:
```bash
python test.py
```
The script will output three standard evaluation metrics:
- **Accuracy**: Classification accuracy for 1-5 influence rating levels
- **MSE**: Mean Squared Error for regression performance
- **MAE**: Mean Absolute Error for prediction error

---

## 📥 Model Checkpoints
| Model | Description | Download Link |
|-------|-------------|---------------|
| Qwen2-VL-7B-Instruct | Base multimodal LLM | [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (downloaded automatically) |
| NetQwen2-VL-7B-final | Fine-tuned full model | [Google Drive]() |

---

## 📝 Citation
If you use this code or dataset in your research, please cite our paper:
```bibtex
@article{xue2026short,
  title={Short-video propagation influence rating: A new real-world dataset and a new large graph model},
  author={Xue, Dizhan and Qian, Shengsheng and Hu, Chuanrui and Xu, Changsheng},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. The Qwen2-VL base model follows its original license.

---

## 🤝 Contact
For questions or issues about the code/dataset, please open an issue or contact [Dizhan Xue](https://github.com/LivXue).
