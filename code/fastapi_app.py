"""FastAPI 推理接口：接收 multipart 图片，调用 TensorRT 推理并返回 JSON。

依赖：fastapi、uvicorn、python-multipart；推理核心复用 code/infer_trt.py。
启动：uvicorn code.fastapi_app:app --host 0.0.0.0 --port 8000
接口：POST /predict?conf=0.25&iou=0.45
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

# 兼容两种运行方式：
# 1) uvicorn code.fastapi_app:app 时走相对导入；
# 2) python code/fastapi_app.py 时走同目录绝对导入。
try:
    from .infer_trt import CLASS_NAMES, TrtRunner
except ImportError:
    from infer_trt import CLASS_NAMES, TrtRunner

# 当前文件在 code/ 下，日志和 engine 都放在仓库根目录。
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
LOG_DIR = REPO_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 默认读取 export/best_fp16.engine；换 engine 时用 ENGINE_PATH 覆盖。
ENGINE_PATH = str(
    Path(os.environ.get("ENGINE_PATH", REPO_DIR / "export" / "best_fp16.engine"))
)

# 标准 logging：同时写 logs/infer.log 和终端，不引入 loguru 等额外依赖。
logger = logging.getLogger("infer_api")
logger.setLevel(logging.INFO)
logger.propagate = False

# 避免 uvicorn reload 或重复 import 时重复挂 handler。
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_DIR / "infer.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# TensorRT 的 engine 和 execution context 只初始化一次，所有请求复用。
_runner: TrtRunner | None = None

# TensorRT context 不能并发调用；用互斥锁把推理串行化。
# 这不是请求队列，也不做并发调度，只是保证同一时刻只有一个请求进 engine。
_infer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启动时加载 engine，退出时释放引用。"""
    global _runner
    try:
        _runner = TrtRunner(ENGINE_PATH)
        logger.info(
            "engine_loaded path=%s input=%s output=%s",
            ENGINE_PATH,
            _runner.input_shape,
            _runner.output_shape,
        )
    except Exception:
        # 启动阶段失败直接让 uvicorn 退出，避免服务起来但每个请求都 500。
        logger.exception("engine_load_failed path=%s", ENGINE_PATH)
        raise

    yield

    _runner = None


app = FastAPI(
    title="YOLOv11n Road Damage Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    """接收一张图片，返回检测框和本次请求耗时。

    file：multipart 图片文件。
    conf：置信度阈值，范围 0~1。
    iou：NMS 的 IoU 阈值，范围 0~1。
    """
    started = time.perf_counter()
    image_name = Path(file.filename or "upload").name

    try:
        # 读取上传内容；空文件直接报错，避免后面解码出 None。
        raw = file.file.read()
        if not raw:
            raise ValueError("empty upload")

        # 用 OpenCV 从内存解码，得到 BGR 图片。
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot decode image")

        # 推理和 NMS 都在 infer_trt.TrtRunner 里，接口层不重复实现。
        with _infer_lock:
            if _runner is None:
                raise RuntimeError("engine not loaded")
            detections = _runner.infer(image, conf=conf, iou=iou)

        # 把 (xyxy, score, cls_id) 转成 JSON 友好的字典。
        boxes = [
            {
                "cls_id": int(cls_id),
                "cls_name": CLASS_NAMES[int(cls_id)],
                "conf": round(float(score), 6),
                "xyxy": [round(float(v), 2) for v in xyxy],
            }
            for xyxy, score, cls_id in detections
        ]

        # elapsed_ms 是每次请求的真实耗时，不做任何预设。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "image=%s conf=%.3f iou=%.3f boxes=%d elapsed_ms=%.2f",
            image_name,
            conf,
            iou,
            len(boxes),
            elapsed_ms,
        )
        return {
            "image_name": image_name,
            "boxes": boxes,
            "elapsed_ms": elapsed_ms,
        }

    except Exception as exc:
        # 任何单次请求异常都返回 500 + 摘要，进程继续服务后续请求。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "request_failed image=%s conf=%.3f iou=%.3f elapsed_ms=%.2f",
            image_name,
            conf,
            iou,
            elapsed_ms,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": f"{type(exc).__name__}: {exc}",
                "image_name": image_name,
                "elapsed_ms": elapsed_ms,
            },
        )


# 启动命令：uvicorn code.fastapi_app:app --host 0.0.0.0 --port 8000
