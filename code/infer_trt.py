"""TensorRT FP16 推理核心：加载 engine、预处理、执行推理、NMS 后处理。

依赖：tensorrt、torch、opencv-python、numpy。
调用方式：
  CLI：python code/infer_trt.py --engine export/best_fp16.engine --image <图片路径>
  API：from infer_trt import TrtRunner
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# TensorRT 的 DLL 默认放在仓库根目录的 tensorrt/lib；也可以用 TRT_LIB 覆盖。
# 必须在 import tensorrt 之前把目录加入 PATH，否则 Windows 会报 DLL 缺失。
TRT_LIB = os.environ.get(
    "TRT_LIB", str(Path(__file__).resolve().parent.parent / "tensorrt" / "lib")
)
_cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin" if os.environ.get("CUDA_PATH") else None
_DLL_HANDLES = []

# 依次加入 TensorRT 和 CUDA 的 bin 目录。
# os.add_dll_directory 的返回值要保留，否则目录句柄可能被回收。
for _dll_dir in (TRT_LIB, str(_cuda_bin) if _cuda_bin else ""):
    if _dll_dir and os.path.isdir(_dll_dir):
        os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(_dll_dir))

import cv2
import numpy as np
import tensorrt as trt
import torch

# 类别顺序必须和训练时一致；输出通道 = 4 个框坐标 + 4 个类别。
CLASS_NAMES = [
    "Longitudinal Crack (D00)",
    "Transverse Crack (D10)",
    "Alligator Crack (D20)",
    "Pothole (D40)",
]

# TensorRT 的 dtype 转成 PyTorch 的 dtype，用于分配 GPU 输入输出缓冲。
TORCH_DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
}
if hasattr(trt, "bool"):
    TORCH_DTYPES[trt.bool] = torch.bool


def letterbox(image: np.ndarray, new_shape: tuple[int, int]):
    """保持宽高比缩放，并用 114 灰色填充到目标尺寸。

    返回：
      canvas：填充后的 BGR 图；
      ratio：缩放比例；
      left/top：左右、上下填充的像素数，后处理要把框映射回原图。
    """
    new_w, new_h = new_shape
    h, w = image.shape[:2]

    # 取较小的缩放比，保证整张图都能放进目标尺寸。
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 剩余空间平均分到两边，得到 padding。
    dw, dh = (new_w - nw) / 2.0, (new_h - nh) / 2.0
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))

    # 先铺满灰色画布，再把缩放后的图贴到中间。
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, left, top


def load_engine(path: str):
    """反序列化本机构建的 TensorRT engine。

    engine 与 GPU 架构、TensorRT 版本绑定，不能直接拷贝到其他机器使用。
    """
    logger = trt.Logger(trt.Logger.WARNING)

    # YOLO 的 ONNX 通常不需要自定义插件，但这里先初始化插件库，兼容性更好。
    trt.init_libnvinfer_plugins(logger, "")

    with open(path, "rb") as f:
        data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        raise RuntimeError(
            f"deserialize_cuda_engine failed: {path}\n"
            "Engine must be built on this machine with the same TensorRT/GPU."
        )

    # runtime 要一直保留，engine 依赖它。
    return engine, runtime


def allocate_buffers(engine, context, imgsz: int):
    """为 engine 的每个 binding 分配 GPU tensor，并返回绑定地址。

    本项目是静态 1×3×640×640 输入。如果换成动态 engine，这里会按 imgsz
    设置输入 shape，然后重新读取输出 shape。
    """
    # 第一遍：处理输入 binding 的动态 shape。
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            continue

        shape = tuple(engine.get_binding_shape(i))
        if len(shape) != 4:
            raise RuntimeError(f"expected 4-D input, got {shape}")

        if any(d < 0 for d in shape):
            shape = (1, 3, imgsz, imgsz)
            if not context.set_binding_shape(i, shape):
                raise RuntimeError(f"set_binding_shape failed for binding {i}")

    buffers: dict[str, torch.Tensor] = {}
    bindings = [0] * engine.num_bindings
    input_indices: list[int] = []
    output_indices: list[int] = []

    # 第二遍：按 binding 的真实 shape/dtype 分配显存。
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(context.get_binding_shape(i))
        if any(d < 0 for d in shape):
            raise RuntimeError(f"unresolved binding shape: {name} {shape}")

        dtype = TORCH_DTYPES.get(engine.get_binding_dtype(i))
        if dtype is None:
            raise TypeError(f"unsupported TensorRT dtype for {name}")

        # torch tensor 的 data_ptr 就是 TensorRT 需要的 device pointer。
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        buffers[name] = tensor
        bindings[i] = int(tensor.data_ptr())
        (input_indices if engine.binding_is_input(i) else output_indices).append(i)

    if len(input_indices) != 1:
        raise RuntimeError(f"expected 1 input, got {len(input_indices)}")
    if not output_indices:
        raise RuntimeError("engine has no output binding")

    return buffers, bindings, input_indices[0], output_indices


def postprocess(
    pred: np.ndarray,
    image_shape: tuple[int, int],
    ratio: float,
    pad_left: int,
    pad_top: int,
    conf_thres: float,
    iou_thres: float,
):
    """把原始输出 (1, 4+nc, N) 解码成检测框。

    步骤：
      1. 去掉 batch 维，统一成 (N, 4+nc)；
      2. 取每个 anchor 的最大类别分数，按 conf 过滤；
      3. 把网络输入坐标的 xywh 映射回原图 xyxy；
      4. 按类别做 NMS，去掉重复框。
    """
    pred = np.asarray(pred, dtype=np.float32)

    # 导出形状通常是 (1, 8, N)，去掉 batch 后变成 (8, N)。
    if pred.ndim == 3:
        pred = pred[0]

    # 兼容 (8, N) 和 (N, 8) 两种排布，统一转成 (N, 8)。
    if pred.shape[0] == 4 + len(CLASS_NAMES):
        pred = pred.T
    elif pred.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(f"unexpected output shape: {pred.shape}")

    # 前 4 列是框，后面是 4 个类别分数。
    scores = pred[:, 4 : 4 + len(CLASS_NAMES)]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(cls)), cls]

    # 先用置信度阈值过滤，减少后续计算。
    mask = conf >= conf_thres
    if not mask.any():
        return []

    # 网络输出是中心点 + 宽高，单位是 640×640 输入图。
    xywh = pred[:, :4].copy()
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2.0 - pad_left) / ratio
    xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2.0 - pad_top) / ratio
    xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2.0 - pad_left) / ratio
    xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2.0 - pad_top) / ratio

    # 去掉 padding 后，框可能超出原图边界，这里裁剪到图像范围内。
    h, w = image_shape
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w - 1)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h - 1)

    # 只保留置信度达标的候选框。
    order = np.where(mask)[0]
    boxes = xyxy[order]
    scores_sel = conf[order]
    cls_sel = cls[order]

    # 过滤掉宽或高小于 1 像素的退化框。
    valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    order, boxes, scores_sel, cls_sel = (
        order[valid],
        boxes[valid],
        scores_sel[valid],
        cls_sel[valid],
    )

    # 按类别分别做 NMS，避免不同类别的框互相抑制。
    kept: list[tuple[int, float, int]] = []
    for c in np.unique(cls_sel):
        idx = np.where(cls_sel == c)[0]
        b = boxes[idx]
        rects = np.stack(
            [b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1
        )
        keep = cv2.dnn.NMSBoxes(
            rects.tolist(),
            scores_sel[idx].tolist(),
            float(conf_thres),
            float(iou_thres),
        )
        if len(keep) == 0:
            continue

        # keep 是当前类别内的局部下标，要映射回原始候选框下标。
        for k in np.asarray(keep).reshape(-1):
            j = int(idx[int(k)])
            kept.append((int(order[j]), float(scores_sel[j]), int(c)))

    # 按置信度从高到低排序，方便前端展示。
    kept.sort(key=lambda item: item[1], reverse=True)
    return [(xyxy[i], score, c) for i, score, c in kept]


class TrtRunner:
    """封装 TensorRT engine、context 和显存缓冲。

    一个进程只创建一个实例即可。调用 infer() 的代码需要自己保证串行，
    例如 FastAPI 里用互斥锁包住，避免多个线程同时用同一个 context。
    """

    def __init__(self, engine_path: str, imgsz: int = 640):
        # engine 和 runtime 必须同时保留，runtime 释放后 engine 可能失效。
        self.engine, self._runtime = load_engine(engine_path)

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed")

        # 输入输出 buffer 只分配一次，后续请求复用。
        self.buffers, self.bindings, self.input_idx, self.output_indices = allocate_buffers(
            self.engine, self.context, imgsz
        )
        self.imgsz = imgsz
        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.output_name = self.engine.get_binding_name(self.output_indices[0])

    @property
    def input_shape(self):
        """返回输入 binding 的 shape，例如 (1, 3, 640, 640)。"""
        return tuple(self.context.get_binding_shape(self.input_idx))

    @property
    def output_shape(self):
        """返回输出 binding 的 shape，例如 (1, 8, 8400)。"""
        return tuple(self.context.get_binding_shape(self.output_indices[0]))

    def io_info(self):
        """返回所有 binding 的信息，CLI 启动时打印出来方便核对。"""
        return [
            (
                self.engine.binding_is_input(i),
                self.engine.get_binding_name(i),
                tuple(self.context.get_binding_shape(i)),
                self.engine.get_binding_dtype(i),
            )
            for i in range(self.engine.num_bindings)
        ]

    def infer(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        """对一张 BGR 图片执行推理，返回 [(xyxy, score, cls_id), ...]。"""
        in_shape = self.input_shape
        net_h, net_w = int(in_shape[2]), int(in_shape[3])

        # 1) letterbox 到网络输入尺寸，并保留映射参数。
        canvas, ratio, pad_left, pad_top = letterbox(image, (net_w, net_h))

        # 2) BGR->RGB、HWC->CHW、归一化到 0~1，并补上 batch 维。
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        # 3) 把数据拷到预先分配好的 GPU 输入 tensor。
        in_tensor = self.buffers[self.input_name]
        in_tensor.copy_(torch.from_numpy(blob).to(device="cuda", dtype=in_tensor.dtype))

        # 4) 在 PyTorch 当前 CUDA stream 上异步执行，然后同步等待完成。
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v2(self.bindings, stream)
        if not ok:
            raise RuntimeError("execute_async_v2 failed")
        torch.cuda.synchronize()

        # 5) 把输出拷回 CPU，检查 NaN/Inf 后做后处理。
        pred = self.buffers[self.output_name].float().cpu().numpy()
        if not np.isfinite(pred).all():
            raise RuntimeError("output contains NaN/Inf")

        return postprocess(pred, image.shape[:2], ratio, pad_left, pad_top, conf, iou)


def main() -> None:
    """CLI 入口：推理一张图片，并把带框结果保存成 jpg。"""
    parser = argparse.ArgumentParser(description="TensorRT YOLOv11n 单图推理")
    parser.add_argument("--engine", required=True, help="TensorRT engine 路径")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="动态 engine 的输入尺寸")
    parser.add_argument("--out", default="", help="结果图路径；默认在输入图旁边")
    args = parser.parse_args()

    # OpenCV 读入的是 BGR 图。
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {args.image}")

    # 初始化 runner，并打印输入输出 binding，方便确认 engine 没接错。
    runner = TrtRunner(args.engine, args.imgsz)
    print(f"TensorRT {trt.__version__}")
    for is_input, name, shape, dtype in runner.io_info():
        print(f"{'IN ' if is_input else 'OUT'} {name} {shape} {dtype}")

    detections = runner.infer(image, conf=args.conf, iou=args.iou)

    # 把检测框画到原图上。
    vis = image.copy()
    for (x1, y1, x2, y2), score, c in detections:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = f"{CLASS_NAMES[c]} {score:.2f}"
        cv2.putText(
            vis,
            label,
            (p1[0], max(p1[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 没指定 --out 时，默认保存为 <原文件名>_trt.jpg。
    out_path = (
        Path(args.out)
        if args.out
        else Path(args.image).with_name(Path(args.image).stem + "_trt.jpg")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), vis):
        raise RuntimeError(f"cannot write image: {out_path}")

    print(f"TRT_OK boxes={len(detections)} out={out_path}")


if __name__ == "__main__":
    main()
