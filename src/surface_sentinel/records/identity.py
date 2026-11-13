"""검사 신원 EngineSpec (P0-2/P0-3, 판정 로드맵 2단계).

평가·API·워커가 **같은** 검사 신원 정의를 공유하기 위한 동결 dataclass.
신원 필드(엔진·task·모델 SHA·설정 해시·정책 해시·전처리/결합 버전·코드 버전)를
canonical JSON 으로 해시한 것이 ``inspection_fingerprint`` 이고, 이미지 SHA 와
쌍으로 UNIQUE 중복제거 키를 이룬다(``records/fingerprint.py``).

``onnx_path`` 는 추적용 참고 필드이며 fingerprint 에 **포함하지 않는다**(경로는
플랫폼 의존이고, 모델 신원은 바이트 SHA 인 ``model_sha256`` 로 고정한다).

``code_sha``/``code_dirty`` 는 진입점(API 생성·워커 시작·평가 CLI)에서
``evaluation.recalibration.git_code_identity`` 로 1회 계산해 주입한다.
이 모듈은 subprocess/네트워크를 쓰지 않는다(워커 핫패스 금지 규칙).
"""

from __future__ import annotations

from dataclasses import dataclass

from .fingerprint import PREPROCESSING_VERSION, inspection_fingerprint


@dataclass(frozen=True)
class EngineSpec:
    """검사 한 건의 신원. 평가·서빙이 같은 값을 써야 짝 일치한다."""

    engine: str
    task: str = ""
    model_sha256: str = ""
    config_hash: str = ""
    yolo_policy_hash: str = ""
    combine_policy_hash: str = ""
    preprocessing_version: str = PREPROCESSING_VERSION
    code_sha: str = ""
    code_dirty: bool = False
    #: 추적용 모델 파일 경로(fingerprint 미포함).
    onnx_path: str = ""

    @property
    def fingerprint(self) -> str:
        """이미지 SHA 를 제외한 신원 필드의 canonical SHA256."""

        return inspection_fingerprint(
            engine=self.engine,
            task=self.task,
            model_sha256=self.model_sha256,
            config_hash=self.config_hash,
            yolo_policy_hash=self.yolo_policy_hash,
            combine_policy_hash=self.combine_policy_hash,
            preprocessing_version=self.preprocessing_version,
            code_sha=self.code_sha,
            code_dirty=self.code_dirty,
        )

    @property
    def provenance_complete(self) -> bool:
        """신원이 완전한가. opencv 는 모델 파일이 없으므로 항상 완전.

        YOLO 계열은 실제 모델 SHA 가 있어야 완전하다(과거 행은 추측하지 않는다).
        """

        return self.engine == "opencv" or bool(self.model_sha256)


def opencv_spec(config_hash: str, *, code_sha: str = "", code_dirty: bool = False) -> EngineSpec:
    """OpenCV 기준선 엔진의 신원(모델 파일·YOLO 정책 없음)."""

    return EngineSpec(
        engine="opencv",
        config_hash=config_hash,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )


__all__ = ["EngineSpec", "opencv_spec"]
