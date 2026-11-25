"""`python -m surface_sentinel` 실행 진입점.

패키지를 모듈 형태로 호출할 때 CLI `main()`으로 위임한다.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
