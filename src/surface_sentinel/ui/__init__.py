"""로컬 웹 검사·검토·관제 UI 패키지.

배치 run 결과와 SQLite 검사 DB를 브라우저에서 조회하고 REVIEW 샘플을 검토하는 앱을 제공한다.
"""

from .app import create_app

__all__ = ["create_app"]