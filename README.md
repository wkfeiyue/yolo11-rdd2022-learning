# YOLOv11 RDD2022 TensorRT

YOLOv11n baseline 在 RDD2022 四类路面病害上的评测与 TensorRT FP16 部署示例。

| 类别 ID | 名称 | 病害类型 |
|---|---|---|
| 0 | D00 | 纵向裂缝 |
| 1 | D10 | 横向裂缝 |
| 2 | D20 | 龟裂 |
| 3 | D40 | 坑洞 |

> 类别顺序被 `code/infer_trt.py` 的类别映射表硬编码依赖。更换数据集或重新训练后，务必同步修改代码中的映射表，否则框上的标签会全部错位。

仓库包含推理代码、ONNX/TRT 导出路径、Gradio 演示和 FastAPI 接口。
**不包含数据集、训练权重、ONNX 和 TensorRT engine**，需自行准备。

## ⚠️ 部署前必读

1. **TensorRT engine 与 GPU 架构 + TensorRT 版本强绑定**。换显卡、升级或降级 TensorRT 后必须重新构建 engine，直接复用旧 engine 会反序列化失败。
2. 本项目导出为**静态尺寸 ONNX（640×640）**，不使用动态轴。trtexec 无需额外传 `--shapes`。
3. 导出 ONNX 时必须 `nms=False`。若改成 `nms=True`，输出张量结构变为后处理结果，当前 `infer_trt.py` 的后处理无法兼容。

## 目录

```text
.
├── code/
│   ├── app.py              # Gradio 演示（PyTorch 后端）
│   ├── fastapi_app.py      # FastAPI POST /predict（TensorRT 后端）
│   ├── infer_trt.py        # TensorRT 推理核心 + CLI 入口
│   └── __init__.py
├── configs/
│   └── rdd2022_china.yaml  # China 测试集验证配置
├── datasets/.gitkeep       # 数据占位
├── models/.gitkeep         # 权重占位
├── export/.gitkeep         # ONNX / engine 输出占位
├── logs/.gitkeep           # FastAPI 日志目录（缺失时请手动创建）
├── requirements.txt
└── README.md
```

## 环境

Python 3.10 + conda。Windows 下用下方的 `.bat` 命令；Linux / macOS 需把 `conda create -p` 改成 `conda create -n`，环境变量改用 `export`。

### Windows

```bat
conda create -p D:\conda_envs\yolo11_env python=3.10 pip -y
conda activate D:\conda_envs\yolo11_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install tensorrt-8.5.3.1-cp310-none-win_amd64.whl

set TRT_LIB=<TensorRT 解压目录>\lib
set PATH=%TRT_LIB%;%PATH%
```

### Linux

```bash
conda create -n yolo11_env python=3.10 pip -y
conda activate yolo11_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install tensorrt-8.5.3.1-cp310-linux_x86_64.whl

export TRT_LIB=${TENSORRT_ROOT}/lib
export LD_LIBRARY_PATH=${TRT_LIB}:$LD_LIBRARY_PATH
```

说明：

- TensorRT wheel 从 NVIDIA 官网下载，路径与 cp 版本按实际填写。
- CUDA、PyTorch、TensorRT、显卡驱动四者版本必须互相匹配，不要随意混用。
- 引擎必须在本机构建。GTX 1650Ti 是 SM 7.5，不支持 Ampere+ 兼容模式，换卡必须重建。

## 数据

RDD2022 从官方仓库下载，解压到 `datasets/`：

```text
datasets/RDD2022_china/test/
├── images/   *.jpg
└── labels/   *.txt
```

`labels` 使用 YOLO 格式：每行 `cls cx cy w h`，坐标归一化到 0~1。
`configs/rdd2022_china.yaml` 的 `path` 指向数据集根目录，换数据目录时只改这一行。

## 权重

`best.pt` 不进仓库。放到 `models/best.pt`，或从 Release 下载。
权重下载：待补充（建议发布到 GitHub Release 并给出直链与 SHA256）。

## 验证 baseline

只验证一个模型：`baseline_china` 在 China 测试集上的表现。把 baseline_china的 `best.pt` 放到 `models/best.pt`。

```bat
yolo val model=models/best.pt data=configs/rdd2022_china.yaml imgsz=640 batch=1 device=0
```

命令输出 mAP50、mAP50-95、Precision、Recall。

## 导出 ONNX

```bat
yolo export model=models/best.pt format=onnx imgsz=640 opset=12 simplify=False nms=False dynamic=False
```

Ultralytics 默认把 ONNX 写到权重同目录：`models/best.onnx`。

> `simplify=True` 需要额外安装 onnx-simplifier，简化后节点会变化，必须重新跑一次推理验证。

## 构建 TensorRT 引擎

```bat
trtexec --onnx=models/best.onnx --saveEngine=export/best_fp16.engine --fp16 --workspace=1024
```

产物：`export/best_fp16.engine`。`trtexec` 来自 TensorRT 包，确认 `PATH` / `LD_LIBRARY_PATH` 能找到它。

## TensorRT 单图推理

`code/infer_trt.py` 是 CLI 入口，文件末尾的 `if __name__ == "__main__": main()` 负责启动。它会加载 engine、执行 letterbox、推理和 NMS，并把结果画到原图上。

```bat
python code/infer_trt.py --engine export/best_fp16.engine --image datasets/RDD2022_china/test/images/China_Drone_000001.jpg --conf 0.25 --iou 0.45 --out export/result.jpg
```

终端会打印 TensorRT 版本、输入输出 binding 和 `TRT_OK boxes=... out=...`。结果图保存到 `--out` 指定位置。

### 三个入口对比

| 入口 | 启动方式 | 推理后端 |
|---|---|---|
| Gradio | `python code/app.py` | PyTorch `models/best.pt`，不经过 TensorRT |
| FastAPI | `uvicorn code.fastapi_app:app ...` | 启动时创建 `TrtRunner`，复用 `export/best_fp16.engine` |
| CLI | `python code/infer_trt.py ...` | 直接加载 `export/best_fp16.engine` |

FastAPI 不单独启动 `infer_trt.py` 进程；`code/fastapi_app.py` 在 lifespan 里调用 `TrtRunner(ENGINE_PATH)`，和 CLI 共用同一套推理代码。

## 启动服务

### Gradio 演示

```bat
python code/app.py
```

默认地址 `http://127.0.0.1:7860`。支持图片上传、手机拍照和 conf 滑块。

### FastAPI 接口

> 首次运行前确认 `logs/` 目录存在，否则日志写入会报路径异常。

```bat
uvicorn code.fastapi_app:app --host 0.0.0.0 --port 8000
```

接口定义：

```text
POST /predict?conf=0.25&iou=0.45
Content-Type: multipart/form-data
file=<image>
```

返回结构：

```json
{
  "image_name": "China_Drone_000001.jpg",
  "boxes": [
    { "cls_id": 0, "cls_name": "D00", "conf": 0.68, "xyxy": [12.5, 45.2, 220.1, 330.4] }
  ],
  "elapsed_ms": 22.3
}
```

curl 测试：

```bash
curl -X POST "http://127.0.0.1:8000/predict?conf=0.25&iou=0.45" -F "file=@./test.jpg"
```

请求日志写 `logs/infer.log`。

## 命令参数说明

### conda / pip

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `-p` | `D:\conda_envs\yolo11_env` | 环境目录变了，后面所有 `${ENV_PYTHON}` 路径要同步改 |
| `--index-url` | `.../whl/cu117` | 换 CUDA 版本会换 torch 轮子，需同时匹配驱动和 TensorRT，不能随意混用 |
| `tensorrt-...whl` | cp310 | Python 版本换了要换对应 cpXXX 轮子，否则装不上 |

### yolo val

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `model` | `models/best.pt` | 换权重就是换模型；类别顺序仍必须是 0~3 |
| `data` | `configs/rdd2022_china.yaml` | 换 YAML 就是换测试集；改 YAML 里的 `path` 才是换数据目录 |
| `imgsz` | `640` | 改大/改小会改变输入分辨率，影响小目标细节和显存；不要和训练/导出尺寸混用 |
| `batch` | `1` | 只影响评测时的显存和速度，不改变 mAP 定义；显存不足就调小 |
| `device` | `0` | 改 `cpu` 可无 GPU 跑但很慢；换其他编号要确认卡存在 |

### yolo export

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `format` | `onnx` | 换格式产物不同；后面的 trtexec 只接 ONNX |
| `imgsz` | `640` | ONNX 输入尺寸会变，trtexec 和 engine 必须同步改 |
| `opset` | `12` | 改高/改低可能影响 TensorRT 8.5 解析，需重新验证 |
| `simplify` | `False` | 改 `True` 会先做图简化，需要额外的 simplifier 依赖；节点可能变化，要重新验证 |
| `nms` | `False` | 改 `True` 会把 NMS 写进图，输出结构变成后处理结果，当前 `infer_trt.py` 不兼容 |
| `dynamic` | `False` | 改 `True` 会产生动态轴；trtexec 要加 shapes，推理侧也要适配 |
| `batch` | `1`（当前命令未写，使用默认） | 改大于 1 会改变输入 batch，当前服务和脚本按单图处理，不建议改 |

### trtexec

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `--onnx` | `models/best.onnx` | 换输入；必须和实际导出路径一致 |
| `--saveEngine` | `export/best_fp16.engine` | 换输出位置；目录不存在会失败 |
| `--fp16` | 开启 | 去掉会构建 FP32 引擎；精度、显存和速度需实测，不能直接假设 |
| `--workspace` | `1024` | 调小可能让某些 tactic 失败，调大占更多显存；不改变模型结构 |

### infer_trt.py

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `--engine` | 必填 | 换 engine；必须本机构建，换卡不能直接复用 |
| `--image` | 必填 | 换输入图片 |
| `--conf` | `0.25` | 调高减少框、可能漏检；调低增加框、可能误检；不需要重建 engine |
| `--iou` | `0.45` | 调高保留更多重叠框；调低合并更多框；不需要重建 engine |
| `--imgsz` | `640` | 只对动态 engine 生效；静态 engine 以 binding 的尺寸为准 |
| `--out` | 输入图旁 `*_trt.jpg` | 换结果图路径；父目录会自动创建 |

### 服务参数

| 参数 | 默认 | 修改后的影响 |
|---|---|---|
| `YOLO_WEIGHTS` | `models/best.pt` | 换 Gradio 加载的权重；类别必须仍是 0~3 |
| `GRADIO_PORT` | `7860` | 换端口；浏览器和手机访问地址要同步 |
| `MAX_SIDE` | `1280` | 调小减少服务端处理量，但可能丢小裂缝；`0` 关闭缩放；不影响手机上传字节 |
| `--host` | `0.0.0.0` | 改 `127.0.0.1` 只允许本机访问 |
| `--port` | `8000` | 换 FastAPI 端口；curl 和前端地址要同步 |
| `conf` / `iou` query | `0.25` / `0.45` | 只影响当前请求，不需要重建 engine |
| `ENGINE_PATH` | `export/best_fp16.engine` | 换 FastAPI 加载的 engine；必须和当前 GPU/TensorRT 匹配 |

## 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 找不到 `trtexec` | TensorRT lib 未加入 PATH | Windows 加 `%TRT_LIB%` 到 `PATH`；Linux 设 `LD_LIBRARY_PATH` |
| `serialize/deserialize` 失败 | engine 非本机 GPU 构建，或 TensorRT 版本不一致 | 用本机重新执行 `trtexec` 构建 |
| 检测框全错或全空 | 导出时开了 `nms=True`；imgsz 不一致；类别顺序错 | 保持 `nms=False`，训练/导出/engine 尺寸统一，核对 0~3 映射 |
| 输出标签名称错位 | 类别映射表与训练数据类别顺序不一致 | 修改 `infer_trt.py` 中的类别映射表 |
| FastAPI 启动报日志路径错误 | `logs/` 目录不存在 | 手动创建 `logs/` 或改用已存在的日志目录 |
| ONNX 解析失败 | opset 过高 / simplify 引入的节点不支持 | `opset=12`，必要时 `simplify=False` 再导出 |
| CUDA / 驱动报错 | PyTorch、TensorRT、驱动版本不匹配 | 对照版本矩阵重新装对应 cuXXX 轮子 |
| 显存不足 / tactic 失败 | `--workspace` 过大或过小 | 在 512~2048 之间调整后重建 engine |

## 部署说明与限制

- `code/infer_trt.py` 复用同一套 letterbox、NMS 和类别映射，CLI 与 FastAPI 共用。
- 引擎绑定本机 GPU 和 TensorRT 版本，不能跨卡复用。
- 本项目为算法部署演示原型，**不建议直接公网投入生产**。当前缺少：
  - 接口鉴权、身份认证、请求限流
  - 多并发安全封装（`TrtRunner` 未做并发保护）
  - 异常捕获与熔断、模型热更新、版本管理、监控告警
  生产化前需在上述方向补齐能力。
