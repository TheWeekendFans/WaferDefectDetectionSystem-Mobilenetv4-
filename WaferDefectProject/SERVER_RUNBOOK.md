# SERVER RUNBOOK（8x 2080Ti）

## 1) 环境策略（结论）
- Python 训练环境与 C++ 编译环境可以共存，但不是同一套依赖管理方式：
  - Python 侧：使用你现有的 `python 3.9.23 GF` 环境安装训练/导出依赖。
  - C++ 侧：依赖系统级 CUDA、TensorRT、OpenCV、CMake、g++。
- 两者共享同一套 NVIDIA 驱动/CUDA/TensorRT 底座。

## 2) Python 训练环境（基于你的 GF 环境）
```bash
cd /path/to/1/WaferDefectProject
python -m pip install -U pip
python -m pip install torch torchvision torchaudio timm tqdm pillow numpy onnx onnxsim pycuda
```

> 说明：请安装与服务器 CUDA 版本匹配的 PyTorch CUDA 轮子（不要 CPU 版）。

## 3) 多卡训练（8 GPU）
```bash
cd /path/to/1/WaferDefectProject
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
  --data-path ../dataset_repo/Wafer_Map_Datasets.npz \
  --epochs 80 \
  --batch-size 64 \
  --num-workers 16
```

- 目标：验证集 Top-1 > 97.2%
- 最优权重输出：`checkpoints/mobilenetv4_best.pth`

## 4) 导出 ONNX
```bash
cd /path/to/1/WaferDefectProject
python export_onnx.py --checkpoint checkpoints/mobilenetv4_best.pth --onnx mobilenetv4.onnx --onnx-sim mobilenetv4_sim.onnx
```

## 5) 构建 TensorRT FP16 引擎
```bash
cd /path/to/1/WaferDefectProject
trtexec --onnx=mobilenetv4_sim.onnx --saveEngine=mobilenetv4.engine --fp16 --workspace=2048
```

## 6) Python 侧延迟验证
```bash
cd /path/to/1/WaferDefectProject
python verify_tensorrt.py --engine mobilenetv4.engine --runs 300 --warmup 50 --fp16
```

## 7) C++ 编译与运行
```bash
cd /path/to/1/WaferDefectProject/cpp_inference
mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=/usr
make -j
./wafer_infer
```

## 8) 指标判定
- 延迟目标：单帧约 1.5ms
- 产线吞吐目标：>120 WPH

## 9) 常见排查
- `torch.cuda.is_available() == False`：安装了 CPU 版 PyTorch 或驱动/CUDA 不匹配。
- `trtexec: command not found`：TensorRT bin 路径未加入 `PATH`。
- CMake 找不到 TensorRT：传 `-DTENSORRT_ROOT=<TensorRT安装路径>`。
