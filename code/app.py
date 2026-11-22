"""Gradio 演示入口：上传图片或手机拍照，拖动 conf 滑块查看检测框。

依赖：ultralytics、gradio、opencv-python。
权重：默认读取仓库根目录下的 models/best.pt，也可以用 YOLO_WEIGHTS 指定。
运行：python code/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 关闭 Gradio 的匿名统计，离线环境更干净。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

# 当前文件在 code/ 下，仓库根目录是它的上一级。
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

# 权重不进 Git，默认从 models/best.pt 读取；也可以用环境变量覆盖。
WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(REPO_DIR / "models" / "best.pt"))

# 模型训练和推理使用 640；MAX_SIDE 只限制上传图片的最长边，减少后续处理量。
IMG_SIZE = 640
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1280"))

# 有 CUDA 就用 0 号卡，没有则退回 CPU；CPU 也能跑，只是慢。
DEVICE = 0 if torch.cuda.is_available() else "cpu"

if not os.path.isfile(WEIGHTS):
    sys.exit(f"WEIGHTS not found: {WEIGHTS}\nSet YOLO_WEIGHTS or edit WEIGHTS.")

# YOLO 对象只加载一次，后续每次请求复用，避免重复读盘和初始化。
model = YOLO(WEIGHTS)


def limit_size(image: np.ndarray, max_side: int) -> np.ndarray:
    """把图片最长边缩到 max_side；max_side <= 0 时关闭缩放。

    只缩小、不放大，保持宽高比，避免把原图拉变形。
    """
    if max_side <= 0:
        return image

    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def predict(image: np.ndarray | None, conf: float):
    """Gradio 回调：接收 RGB 图片和 conf，返回带框图片与文字明细。"""
    if image is None:
        return None, "请先上传图片或用手机拍照。"

    # 先限制尺寸，避免手机原图过大导致后续推理和结果图都很慢。
    image = limit_size(image, MAX_SIDE)

    # Gradio 给的是 RGB，Ultralytics/OpenCV 习惯 BGR，这里先换通道。
    bgr = np.ascontiguousarray(image[:, :, ::-1])

    # imgsz 是模型输入尺寸，conf 由滑块控制，verbose=False 避免刷屏。
    result = model.predict(
        source=bgr,
        imgsz=IMG_SIZE,
        conf=float(conf),
        device=DEVICE,
        verbose=False,
    )[0]

    # result.plot() 返回 BGR 标注图，转回 RGB 给 Gradio 显示。
    vis = result.plot()[:, :, ::-1]

    # 把检测框整理成可读文本，方便在页面上核对类别、置信度和坐标。
    lines: list[str] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            name = result.names.get(cls_id, str(cls_id))
            lines.append(
                f"{name}  conf={score:.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )

    if not lines:
        lines = ["未检测到目标；可下调 conf 后重试。"]

    return vis, "\n".join(lines)


# 页面布局：一张输入图片、一个 conf 滑块、一张结果图和一块文字明细。
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(
            type="numpy",
            sources=["upload", "webcam"],
            label="上传图片 / 手机拍照",
        ),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="conf 阈值"),
    ],
    outputs=[
        gr.Image(type="numpy", label="检测结果"),
        gr.Textbox(label="检测明细", lines=6),
    ],
    title="YOLOv11n（baseline）路面病害检测",
    description=(
        "4 类：D00 纵向裂缝 / D10 横向裂缝 / D20 龟裂 / D40 坑洞。"
        "手机摄像头需 HTTPS 页面；HTTP 局域网地址只能上传图片。"
        f"输入图最长边限制为 {MAX_SIDE} 像素（MAX_SIDE=0 关闭）。"
    ),
)

if __name__ == "__main__":
    # 0.0.0.0 让同一局域网内的手机也能访问；端口可用 GRADIO_PORT 覆盖。
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        inbrowser=True,
    )
