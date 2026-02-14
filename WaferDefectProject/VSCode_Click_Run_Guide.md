# WaferDefectProject：VS Code 点击运行指南（不手敲终端版）

这份文档按“点按钮运行”的方式来做，不要求你手动输入终端命令。

---

## 0. 一次性准备（只做一次）

### 0.1 打开项目
1. 在 VS Code 中点击 File -> Open Folder。
2. 选择 1/WaferDefectProject。

### 0.2 选择 Python 解释器（你的 GF 环境）
1. 按 Ctrl+Shift+P。
2. 输入并选择 Python: Select Interpreter。
3. 选择你成熟的 Python 3.9.23 GF 环境。
4. 看 VS Code 右下角，确认显示该环境。

### 0.3 安装必要扩展（如果未安装）
1. 左侧扩展商店安装 Python（Microsoft）。
2. 左侧扩展商店安装 CMake Tools（Microsoft）。
3. 左侧扩展商店安装 C/C++（Microsoft）。

### 0.4 你不会配 C++ 时，直接用这个脚本
1. 打开 `cpp_inference/build_and_run.ps1`。
2. 点击右上角运行（Run PowerShell File）。
3. 它会自动：
   - 查找 TensorRT SDK（`NvInfer.h` + `nvinfer.lib`）
   - 查找 OpenCV
   - 配置并编译 C++
   - 编译成功后自动运行

---

## 1. 训练模型（点击运行 train.py）

### 你要点哪里
1. 在资源管理器打开 train.py。
2. 点击编辑器右上角 Run Python File 按钮（绿色三角形）。

### 你会看到什么
1. 终端面板会自动弹出并开始训练日志（你不用手打命令）。
2. 日志会显示：
   - Using device: cuda
   - GPUs: 8（如果 8 张卡都可见）
   - Epoch xxx | Train Acc xx.xx% | Val Acc xx.xx%
3. 每当验证集更高，会打印 Saved Best Model!。

### 这一步产出什么
1. checkpoints/mobilenetv4_best.pth（最佳权重）
2. checkpoints/mobilenetv4_final.pth（最终权重）

### 达标标准
- 目标是 Val Acc（Top-1）> 97.2%。

---

## 2. 导出 ONNX（点击运行 export_onnx.py）

### 你要点哪里
1. 打开 export_onnx.py。
2. 点击右上角 Run Python File。

### 你会看到什么
1. Loading weights from checkpoints/mobilenetv4_best.pth
2. ONNX Export successful.
3. Simplified model saved to mobilenetv4_sim.onnx

### 这一步产出什么
1. mobilenetv4.onnx
2. mobilenetv4_sim.onnx（优先用于 TensorRT）

---

## 3. TensorRT 验证（点击运行 verify_tensorrt.py）

### 你要点哪里
1. 打开 verify_tensorrt.py。
2. 点击右上角 Run Python File。

### 你会看到什么
1. 若本目录还没有 engine，会自动构建：
   - Building TensorRT Engine...
   - FP16 support detected. Enabling FP16.
2. 然后开始推理基准测试：
   - TensorRT Inference Latency: x.xx ms

### 这一步产出什么
1. mobilenetv4.engine（若此前不存在）
2. 一条延迟结果日志（目标约 1.5ms）

---

## 4. C++ 编译与运行（通过 VS Code 按钮）

> 不想手动配置时，优先使用 `cpp_inference/build_and_run.ps1` 一键方式。

### 4.0 先确认 3 个前置条件（只做一次）
1. 已安装 **Visual Studio 2019/2022 + Desktop development with C++**。
2. 已安装 **CUDA Toolkit（含 Visual Studio Integration）**。
3. 已安装 **TensorRT C++ SDK（开发版）**，并能找到 `NvInfer.h` 和 `nvinfer.lib`。

> 注意：仅安装 Python 的 `tensorrt` 包不够，它不提供 C++ 编译所需开发头文件。

### 4.1 在 VS Code 里设置 CMake 变量（点击操作）
1. Ctrl+Shift+P -> **CMake: Edit User-Local CMake Kits**（或打开 CMake Settings UI）。
2. 配置以下变量：
   - `TENSORRT_ROOT` = 你的 TensorRT SDK 根目录（例如 `C:/TensorRT-10.x.x.x`）
   - `OpenCV_DIR` = `C:/Users/zpl/.conda/pkgs/opencv-4.10.0-py39hd762f8c_0/Library/cmake`
3. 如果之前配置失败过，执行：Ctrl+Shift+P -> **CMake: Delete Cache and Reconfigure**。

### 你要点哪里
1. 在 VS Code 打开文件夹 1/WaferDefectProject/cpp_inference。
2. 左下角状态栏会出现 CMake 按钮。
3. 点击 **CMake: Select Kit**，选择 Visual Studio x64 Kit。
4. 点击 **CMake: Configure**。
5. 点击 **CMake: Build**（构建按钮）。
6. 构建成功后，点击状态栏 **Run**（或 CMake: Run Without Debugging）。

### 你会看到什么
1. Build succeeded 或 100% Built target wafer_infer。
2. 运行输出中会显示：
   - Predicted Class: xx
   - Avg Latency: x.xx ms

### 这一步产出什么
1. cpp_inference/build/wafer_infer（可执行文件）
2. C++ 端平均延迟结果

### 4.2 engine 文件位置说明（已帮你优化）
- `wafer_infer` 现在会自动在当前目录及上级目录寻找 `mobilenetv4.engine`，通常不需要手动复制。
- 若你想指定路径，也可以在运行参数里传 engine 路径作为第一个参数。

---

## 5. 你只看这 4 个关键结果就行

1. 训练准确率：Val Acc > 97.2%
2. 最优权重：checkpoints/mobilenetv4_best.pth
3. TensorRT 引擎：mobilenetv4.engine
4. 延迟结果：Python TRT 与 C++ 推理都接近 1.5ms（设备相关会有波动）

---

## 6. 常见问题（点击运行时）

### 问题 A：train.py 显示 CUDA False
- 原因：解释器没选到你的 GF 环境，或该环境装的是 CPU 版 torch。
- 处理：重新 Python: Select Interpreter，确认右下角环境名正确。

### 问题 B：verify_tensorrt.py 报找不到 tensorrt/pycuda
- 原因：GF 环境里没装对应包，或 TensorRT Python 绑定不在该环境。
- 处理：让管理员在 GF 环境补齐 TensorRT Python 绑定与 pycuda。

### 问题 C：CMake 找不到 TensorRT
- 原因：系统 TensorRT 开发库路径未配置。
- 处理：在 VS Code 的 CMake Tools 配置里设置 TENSORRT_ROOT 到服务器实际安装目录。

### 问题 D：报错 No CUDA toolset found
- 原因：CUDA 没有安装 VS 集成组件，或 VS C++ 工具链不完整。
- 处理：重装/修复 CUDA Toolkit，勾选 Visual Studio Integration；并确认 VS 安装了 Desktop development with C++。

---

## 7. 推荐你的实际操作顺序（最省事）

1. 先跑 train.py（拿到 best.pth）
2. 再跑 export_onnx.py（拿到 onnx）
3. 再跑 verify_tensorrt.py（拿到 engine 和 Python 延迟）
4. 最后用 CMake 按钮编译运行 C++（拿到 C++ 延迟）

这套顺序最稳，也最适合写入简历项目过程。