"""OpenCV r20 기준선과 비교·결합하기 위한 YOLO(det/seg) 학습·ONNX 추론·평가 패키지.

학습은 격리 환경(`.venv-train`, torch CPU + ultralytics)에서 수행하고,
제품 추론은 ONNX Runtime과 자체 결정론적 후처리로 수행한다.
기존 OpenCV 파이프라인은 기준선으로 보존한다.
"""
