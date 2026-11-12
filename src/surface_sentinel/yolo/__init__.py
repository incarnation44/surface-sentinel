"""YOLO 비교 트랙(ADR 0007).

학습은 격리 환경(`.venv-train`, torch CPU + ultralytics)에서 수행하고,
제품 추론은 ONNX Runtime과 자체 결정론적 후처리로 수행한다.
기존 OpenCV 파이프라인은 기준선으로 보존한다.
"""