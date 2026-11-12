"""YOLO 학습·ONNX export 스크립트(격리 환경 `.venv-train` 전용, ADR 0007 D2).

이 스크립트는 ultralytics만 사용하며 ``surface_sentinel`` 패키지를 import하지
않는다. 메인 venv에 torch/ultralytics를 넣지 않기 위한 격리 학습 진입점이다.

실행 예(저장소 루트에서):
    .venv-train\\Scripts\\python.exe src\\surface_sentinel\\yolo\\train.py --task det
    .venv-train\\Scripts\\python.exe src\\surface_sentinel\\yolo\\train.py --task seg

학습 산출물(best.pt, best.onnx, results.csv)은 ``data/derived/yolo/models/<task>/``
아래에 저장되며 Git에 추가하지 않는다.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# ultralytics가 누락 의존성을 임의 설치하지 않도록 한다(재현성).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# 분석 텔레메트리 비활성화(외부 요청 없음).
os.environ.setdefault("ULTRALYTICS_SYNC", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402

DEFAULT_SEED = 20261112
MODELS = {"det": "yolov8n.pt", "seg": "yolov8n-seg.pt"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_and_export(args: argparse.Namespace) -> Path:
    SETTINGS["sync"] = False
    _seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    data = Path(args.data).resolve()
    project = Path(args.project).resolve()
    model = YOLO(MODELS[args.task])
    model.train(
        data=str(data),
        task="segment" if args.task == "seg" else "detect",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=str(project),
        name=args.task,
        exist_ok=True,
        cache=False,
        plots=True,
        val=True,
        verbose=True,
    )
    best = project / args.task / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"학습 가중치가 없습니다: {best}")

    exported = YOLO(str(best)).export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=False,
        half=False,
        device="cpu",
    )
    print(f"ONNX export: {exported}")
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO CPU 학습 + ONNX export(격리 환경).")
    parser.add_argument("--task", choices=("det", "seg"), required=True)
    parser.add_argument(
        "--data",
        default=None,
        help="dataset.yaml 경로 (기본 data/derived/yolo/<task>/dataset.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--project", default="data/derived/yolo/models")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.data is None:
        args.data = f"data/derived/yolo/{args.task}/dataset.yaml"
    best = train_and_export(args)
    print(f"학습 완료: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())