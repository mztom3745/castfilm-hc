from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def iter_image_files(input_dir: Path) -> List[Path]:
    """Return all image files inside the given directory (non-recursive)."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No image files found inside: {input_dir}")
    return files


def ensure_directory(path: Path) -> None:
    """Create the directory when missing."""
    path.mkdir(parents=True, exist_ok=True)


def write_report(report_path: Path, sections: Iterable[str]) -> None:
    """Persist the detection report with newline separation."""
    ensure_directory(report_path.parent)
    content = "\n".join(sections)
    report_path.write_text(content, encoding="utf-8")
