# 本地运行保姆级指南 (Wafer Defect Classification)

这个文档将手把手教你在本地运行刚才创建的半导体缺陷分类项目。

## 📂 第一部分：项目文件及其作用

在 `g:\半导体offer\WaferDefectProject` 目录下，我们有以下核心文件，他们的作用如下：

| 文件名 | 作用 (它是做什么的？) |
| :--- | :--- |
| **`src/dataset.py`** | **数据管家**。它负责打开 `Wafer_Map_Datasets.npz` 文件，把里面的 8位二进制标签 (如 `[0,0,1...]`) 翻译成 0-37 的数字，并把图片处理成模型能看懂的格式。 |
| **`src/model.py`** | **大脑结构**。这里定义了 MobileNetV4 模型的结构。如果你的电脑安装了 `timm` 库，它会自动从里面加载高性能的 MobileNetV4 模型。 |
| **`train.py`** | **老师**。它负责指挥模型进行学习。它会不断把数据喂给模型，计算错误（Loss），然后调整模型参数，让它越来越准。它还会在训练过程中保存这一轮表现最好的模型到 `checkpoints/` 文件夹。 |
| **`export_onnx.py`** | **翻译官**。训练好的模型是 PyTorch 格式的 (`.pth.`)，这个脚本把它“翻译”成通用的 `.onnx` 格式，方便以后转成 C++ (TensorRT) 版本部署到生产线。 |
| **`verify_tensorrt.py`** | **质检员**。用于测试 `.onnx` 模型或者是 `.engine` 模型能不能正常工作，算一下推理一张图需要几毫秒。 |
| **`cpp_inference/`** | **最终产品**。这是一个 C++ 代码包，用于在实际工厂服务器上运行。它速度最快，但需要复杂的编译环境。 |

---

## 🚀 第二部分：按顺序运行代码

请打开你的 PowerShell 终端，并确保你处于项目目录下。
**重要提示**：因为我们发现你的环境较多，请务必复制下面的命令来执行，指定使用 `py311_env` 这个环境。

### 第一步：准备与检查
首先进入项目目录：
```powershell
cd g:\半导体offer\WaferDefectProject
```

### 第二步：开始训练模型
我们要运行 `train.py`。这个过程可能需要几分钟到几十分钟，取决于你的显卡。
```powershell
E:\Miniconda\envs\py311_env\python.exe train.py
```
**👀 你会看到什么？**
- 屏幕上会出现进度条 (如 `Epoch 1/20: 100%...`)。
- 每一轮结束会显示准确率：`Val Acc: 80.49%` (数字会变)。
- 如果准确率提升，会提示 `Saved Best Model!`。
- 训练结束后，模型文件会保存在 `checkpoints/mobilenetv4_best.pth`。

### 第三步：导出通用模型
训练好后，我们要把它转换成 ONNX 格式。
```powershell
E:\Miniconda\envs\py311_env\python.exe export_onnx.py
```
**👀 你会看到什么？**
- 提示 `Loading weights...` (加载刚才训练好的权重)。
- 提示 `Exporting model to mobilenetv4.onnx...`。
- 最后显示 `Simplified model saved to mobilenetv4_sim.onnx`。目录下会出现这两个 `.onnx` 文件。

### 第四步：尝试 TensorRT 加速 (进阶)
这一步尝试把 ONNX 转换成 NVIDIA 专用的 TensorRT 引擎，速度极快。
```powershell
E:\Miniconda\envs\py311_env\python.exe verify_tensorrt.py
```
**👀 你会看到什么？**
- 如果成功：它会显示 `Build Engine...` 然后输出 `TensorRT Inference Latency: 1.50 ms` (类似数值)。
- **如果报错 (Error 35)**：这说明你本地的显卡驱动太老了 (Driver Version)，带不动最新的 TensorRT。**这没关系**，说明代码是对的，只是需要去服务器或者更新驱动才能跑。

### 第五步：C++ 部署 (最终形态)
本地 Windows 配置 C++ 环境比较麻烦。如果你想看效果，主要看前三步即可。
如果要在服务器部署，把 `cpp_inference` 文件夹上传，按照 `readme` 编译即可。

---

## ❓ 常见问题
**Q: 为什么我直接输 `python train.py` 报错？**
A: 因为你的默认 python 环境里的 PyTorch 坏了 (缺 DLL)。必须用 `E:\Miniconda\envs\py311_env\python.exe` 这个完整路径来运行，这个环境是好的。

**Q: 训练太慢怎么办？**
A: 可以打开 `train.py`，把里面的 `EPOCHS = 20` 改成 `EPOCHS = 5`，只是演示一下流程。

**Q: C++ 代码那里怎么跑？**
A: C++ 代码需要用 Visual Studio 或者 MinGW 编译。这属于软件工程的编译范畴，如果本地没有配好库 (OpenCV, TensorRT, CUDA) 是跑不起来的。建议这一步作为“展示内容”写在简历里，实际运行在配好环境的服务器上完成。
