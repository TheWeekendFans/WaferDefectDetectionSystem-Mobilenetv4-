# Wafer Defect Detection System (基于MobileNetV4的晶圆缺陷检测系统)

High-performance wafer map defect detection system deployed on Edge GPU, achieving **98.08% Top-1 Accuracy** and **<1ms Inference Latency**.

## Result Show（结果展示）
<img width="2286" height="948" alt="晶圆缺陷检测随机10张对比图" src="https://github.com/user-attachments/assets/5973e41f-4102-465d-8bb0-1d2223e5f713" />

## Key Features (核心亮点)
- **Model**: MobileNetV4 with custom optimizations.
- **Performance**:
  - **Accuracy**: 98.08% on MixedWM38 dataset (SOTA Level).
  - **Latency**: 0.90ms per image (TensorRT FP16).
  - **Throughput**: Supports >1000 WPH production lines.
- **Deployment**: Full C++ Inference Engine with custom CUDA Preprocessing Kernels.

## Project Structure (项目结构)
- `train.py`: PyTorch training script with mixed precision & cosine annealing.
- `export_onnx.py`: Converter from PyTorch to ONNX with dynamic batch support.
- `cpp_inference/`: C++ TensorRT implementation.
  - `main.cpp`: Inference engine wrapper.
  - `preprocess.cu`: Custom CUDA kernel for high-speed normalization.
- `src/`: Model definition and dataset loader.

## Quick Start
### 1. Training
```bash
python train.py --data-path /path/to/Wafer_Map_Datasets.npz
```

### 2. Export to ONNX
```bash
python export_onnx.py --checkpoint checkpoints/mobilenetv4_best.pth
```

## Requirements
- PyTorch 2.x
- TensorRT 8.6+
- CUDA 11.8 / 12.x
- OpenCV (C++)
