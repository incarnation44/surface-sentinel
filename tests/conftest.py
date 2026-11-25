"""pytest 전역 설정 파일.

테스트 실행 시 `tests/` 루트의 공용 헬퍼 모듈을 import할 수 있도록 환경을 맞춘다.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.append(str(TESTS))
