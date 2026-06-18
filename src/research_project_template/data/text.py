from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def ensure_text_file(
    path: Path, download_url: str | None, allow_download: bool
) -> None:
    if path.exists():
        return
    if not allow_download or download_url is None:
        raise FileNotFoundError(
            f"Could not find {path}. Set data.allow_download=true or provide a file."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(download_url, timeout=30) as response:
        path.write_text(response.read().decode("utf-8"), encoding="utf-8")


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError(f"Expected at least two characters in {path}.")
    return text


def split_text(text: str, train_split: float) -> tuple[str, str]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("data.train_split must be between 0 and 1.")

    split_index = int(len(text) * train_split)
    return text[:split_index], text[split_index:]
