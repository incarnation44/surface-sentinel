"""디코드 전 이미지 헤더 해상도 검사 가드.

압축률이 높은 이미지(예: 200KB PNG → 192MB+)가 ``cv2.imdecode``/``cv2.imread``
단계에서 메모리를 폭주(OOM)시키기 전에, 파일 앞부분 헤더만 읽어 가로·세로와
총 픽셀 상한을 검증한다. ``configs/base.toml``(동결 대상)은 건드리지 않는다.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: 기본 허용 최대 픽셀 수(2,500만 픽셀 ≈ 4채널 uint8 기준 약 100MB).
MAX_IMAGE_PIXELS = 25_000_000

#: 기본 허용 최대 한 변 길이(비정상 극단 종횡비 방어).
MAX_IMAGE_DIMENSION = 16_384

#: 디스크에서 헤더를 읽을 최대 바이트 수(256 KiB; EXIF가 큰 JPEG 포함).
_HEADER_PROBE_BYTES = 262_144

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_JPEG_STANDALONE_MARKERS = frozenset({0x00, 0x01, *range(0xD0, 0xDA)})


class ImageHeaderError(ValueError):
    """지원하지 않거나 손상된 이미지 헤더로 디코드 전 치수를 확인할 수 없을 때 발생한다."""


class ImagePixelLimitError(ValueError):
    """이미지 해상도가 허용 픽셀/변 길이 상한을 초과했을 때 발생한다."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        max_pixels: int,
        max_dimension: int,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_pixels = int(max_pixels)
        self.max_dimension = int(max_dimension)
        super().__init__(
            f"이미지 해상도({self.width}x{self.height})가 허용 상한"
            f"({self.max_pixels:,}px, 최대 변 {self.max_dimension:,}px)을 초과했습니다."
        )


def _probe_png(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _probe_jpeg(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    offset = 2
    size = len(data)
    while offset < size:
        while offset < size and data[offset] != 0xFF:
            offset += 1
        while offset < size and data[offset] == 0xFF:
            offset += 1
        if offset >= size:
            return None
        marker = data[offset]
        offset += 1
        if marker in (0xD9, 0xDA):  # EOI / SOS
            return None
        if marker in _JPEG_STANDALONE_MARKERS:
            continue
        if offset + 2 > size:
            return None
        seg_len = struct.unpack(">H", data[offset : offset + 2])[0]
        if seg_len < 2:
            return None
        if marker in _JPEG_SOF_MARKERS:
            if seg_len < 7 or offset + 7 > size:
                return None
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return int(width), int(height)
        offset += seg_len
    return None


def _probe_bmp(data: bytes) -> tuple[int, int] | None:
    if len(data) < 26 or data[:2] != b"BM":
        return None
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size == 12:
        width, height = struct.unpack("<HH", data[18:22])
        if width <= 0 or height == 0:
            return None
        return int(width), int(height)
    if header_size >= 40:
        width, height = struct.unpack("<ii", data[18:26])
        if width <= 0 or height == 0:
            return None
        return int(width), abs(int(height))
    return None


def _probe_tiff(data: bytes) -> tuple[int, int] | None:
    if len(data) < 8:
        return None
    if data[:4] == b"II*\x00":
        endian = "<"
    elif data[:4] == b"MM\x00*":
        endian = ">"
    else:
        return None
    ifd_offset = struct.unpack(f"{endian}I", data[4:8])[0]
    if ifd_offset + 2 > len(data):
        return None
    entry_count = struct.unpack(f"{endian}H", data[ifd_offset : ifd_offset + 2])[0]
    pos = ifd_offset + 2
    width: int | None = None
    height: int | None = None
    for _ in range(min(entry_count, 256)):
        if pos + 12 > len(data):
            break
        tag, field_type, count = struct.unpack(f"{endian}HHI", data[pos : pos + 8])
        raw_val = data[pos + 8 : pos + 12]
        pos += 12
        if tag not in (256, 257) or count < 1:
            continue
        if field_type == 3:  # SHORT
            val = struct.unpack(f"{endian}H", raw_val[:2])[0]
        elif field_type == 4:  # LONG
            val = struct.unpack(f"{endian}I", raw_val)[0]
        else:
            continue
        if tag == 256:
            width = int(val)
        elif tag == 257:
            height = int(val)
        if width is not None and height is not None:
            return width, height
    return None


def _probe_gif(data: bytes) -> tuple[int, int] | None:
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return int(width), int(height)


def _probe_webp(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return int(width), int(height)
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = struct.unpack("<I", data[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return int(width), int(height)
    return None


def _probe_netpbm(data: bytes) -> tuple[int, int] | None:
    """Netpbm(P1~P6: PBM/PGM/PPM) 헤더 치수를 파싱한다."""

    if len(data) < 7 or data[0:1] != b"P" or data[1:2] not in b"123456":
        return None
    if data[2:3] not in b" \t\r\n#":
        return None
    tokens: list[bytes] = []
    idx = 2
    limit = min(len(data), 1024)
    while idx < limit and len(tokens) < 2:
        ch = data[idx : idx + 1]
        if ch in b" \t\r\n":
            idx += 1
            continue
        if ch == b"#":
            nl = data.find(b"\n", idx, limit)
            if nl == -1:
                return None
            idx = nl + 1
            continue
        start = idx
        while idx < limit and data[idx : idx + 1] not in b" \t\r\n#":
            idx += 1
        tokens.append(data[start:idx])
    if len(tokens) < 2:
        return None
    try:
        width = int(tokens[0].decode("ascii"))
        height = int(tokens[1].decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def probe_image_dimensions(data: bytes) -> tuple[int, int] | None:
    """바이트 앞부분 헤더에서 ``(width, height)``를 추출한다. 판별 불가면 ``None``."""

    if len(data) < 7:
        return None
    for parser in (
        _probe_png,
        _probe_jpeg,
        _probe_bmp,
        _probe_tiff,
        _probe_gif,
        _probe_webp,
        _probe_netpbm,
    ):
        dims = parser(data)
        if dims is not None:
            if dims[0] <= 0 or dims[1] <= 0:
                return None
            return dims
    return None


def probe_image_file_dimensions(path: str | Path) -> tuple[int, int] | None:
    """파일 전체를 디코드하지 않고 헤더 구간만 읽어 ``(width, height)``를 반환한다."""

    with Path(path).open("rb") as handle:
        head = handle.read(_HEADER_PROBE_BYTES)
    dims = probe_image_dimensions(head)
    if dims is not None:
        return dims
    # TIFF IFD가 파일 끝에 놓인 경우만 제한적으로 추가 탐색한다.
    if len(head) >= 8 and head[:4] in (b"II*\x00", b"MM\x00*"):
        endian = "<" if head[:2] == b"II" else ">"
        ifd_offset = struct.unpack(f"{endian}I", head[4:8])[0]
        if ifd_offset >= _HEADER_PROBE_BYTES:
            with Path(path).open("rb") as handle:
                handle.seek(ifd_offset)
                ifd_chunk = handle.read(4096)
            if len(ifd_chunk) >= 2:
                # 가짜 8바이트 TIFF 헤더 + 읽은 IFD 블록으로 재파싱한다.
                synthetic = head[:4] + struct.pack(f"{endian}I", 8) + ifd_chunk
                tiff_dims = _probe_tiff(synthetic)
                if tiff_dims is None or tiff_dims[0] <= 0 or tiff_dims[1] <= 0:
                    return None
                return tiff_dims
    return None


def check_decoded_pixel_budget(
    width: int,
    height: int,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> None:
    """가로·세로가 허용 픽셀/최대 변 상한 이내인지 검증한다."""

    if width <= 0 or height <= 0:
        raise ImageHeaderError(f"유효하지 않은 이미지 치수입니다: {width}x{height}")
    if width > max_dimension or height > max_dimension or width * height > max_pixels:
        raise ImagePixelLimitError(
            width,
            height,
            max_pixels=max_pixels,
            max_dimension=max_dimension,
        )


def check_image_pixel_budget(
    source: str | Path | bytes | bytearray | memoryview,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    require_dimensions: bool = True,
) -> tuple[int, int] | None:
    """디코드 전에 이미지 헤더를 검사해 픽셀 상한을 넘으면 :class:`ImagePixelLimitError`.

    ``require_dimensions=True``(기본)면 헤더에서 치수를 읽지 못할 때 디코드를
    시도하지 않고 :class:`ImageHeaderError` 로 즉시 차단한다(fail-closed).
    """

    if isinstance(source, (bytes, bytearray, memoryview)):
        dims = probe_image_dimensions(bytes(source[:_HEADER_PROBE_BYTES]))
    else:
        dims = probe_image_file_dimensions(source)
    if dims is None:
        if require_dimensions:
            raise ImageHeaderError("지원하지 않거나 손상된 이미지 헤더입니다.")
        return None
    check_decoded_pixel_budget(
        dims[0],
        dims[1],
        max_pixels=max_pixels,
        max_dimension=max_dimension,
    )
    return dims


__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_PIXELS",
    "ImageHeaderError",
    "ImagePixelLimitError",
    "check_decoded_pixel_budget",
    "check_image_pixel_budget",
    "probe_image_dimensions",
    "probe_image_file_dimensions",
]
