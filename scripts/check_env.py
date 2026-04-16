"""Проверка окружения перед запуском бота на сервере."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _fail(msg: str) -> None:
    print(f"[!!] {msg}")


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    tg = importlib.util.find_spec("telegram")
    if tg:
        _ok("python-telegram-bot импортируется")
    else:
        _fail("не найден модуль telegram — pip install -r requirements.txt")
        return 1

    pil = importlib.util.find_spec("PIL")
    if pil:
        _ok("Pillow импортируется")
    else:
        _fail("не найден PIL — pip install -r requirements.txt")
        return 1

    cfg = ROOT / "config.py"
    if not cfg.exists():
        _fail("нет config.py — скопируй из примера и вставь BOT_TOKEN")
    else:
        try:
            text = cfg.read_text(encoding="utf-8")
            if "PASTE_YOUR_TOKEN_HERE" in text or 'BOT_TOKEN = ""' in text:
                _fail("config.py: BOT_TOKEN не задан")
            else:
                _ok("config.py найден, BOT_TOKEN не плейсхолдер")
        except OSError as e:
            _fail(f"config.py: {e}")

    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        _fail("ffmpeg не найден в PATH — установи ffmpeg для обработки видео")
        return 1
    if r.returncode == 0:
        line = (r.stdout or "").splitlines()[0] if r.stdout else "ffmpeg"
        _ok(f"ffmpeg: {line}")
    else:
        _fail("ffmpeg: ошибка запуска")

    try:
        r2 = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        _fail("ffprobe не найден в PATH — нужен вместе с ffmpeg")
        return 1
    if r2.returncode == 0:
        line = (r2.stdout or "").splitlines()[0] if r2.stdout else "ffprobe"
        _ok(f"ffprobe: {line}")
    else:
        _fail("ffprobe: ошибка запуска")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
