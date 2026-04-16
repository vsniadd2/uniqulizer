from __future__ import annotations

import hashlib
import io
import logging
import random
import secrets
import re
import shutil
import subprocess
import sys
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path

import piexif
from PIL import Image, ImageCms, ImageEnhance, ImageFilter, ImageOps

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

# Временно: «динамика» для видео отключена (визуальные/аудио микро-правки).
# Оставляем только стерильную пересборку контейнера (метаданные/таймкод/потоки).
VIDEO_DYNAMICS_TEMPORARILY_DISABLED = True


def _register_optional_image_openers() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    try:
        import pillow_avif  # noqa: F401 — регистрация AVIF в Pillow
    except ImportError:
        pass


_register_optional_image_openers()

# Расширения входа: выход JPEG — декодирование в RGB сбрасывает XMP/IPTC/вложения редакторов из пиксельного контура;
# промежуточное сохранение без чужого EXIF; финал: piexif (Apple/Samsung, даты «недавно», без GPS) + thumbnail в EXIF + sRGB ICC.
IMAGE_INPUT_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
    }
)
VIDEO_INPUT_SUFFIXES = frozenset(
    {
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
        ".mkv",
        ".avi",
        ".mpeg",
        ".mpg",
        ".m2ts",
        ".ts",
        ".flv",
        ".wmv",
        ".3gp",
        ".ogv",
    }
)


def bytes_sha256(data: bytes) -> str:
    """Статлесс-хэш в памяти (без хранения истории)."""
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def _srgb_iec61966_icc_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def document_suffix_from_name(name: str | None, fallback: str = ".bin") -> str:
    if not name:
        return fallback
    n = name.lower()
    for ext in sorted(IMAGE_INPUT_SUFFIXES | VIDEO_INPUT_SUFFIXES, key=len, reverse=True):
        if n.endswith(ext):
            return ext
    return fallback


@dataclass(frozen=True)
class UniqulizeOptions:
    """preset: fb — мягче и меньше артефактов; strong — агрессивнее.
    preserve_visual: для fb — без обрезки/геометрии/заметных цветовых правок: новый файл и битстрим,
    визуально как оригинал; базово EXIF/XMP/ICC снимаются при save.
    synthetic_exif: после чистого JPEG вшить эмуляцию устройства (piexif: Make/Model iPhone/Samsung, недавние даты,
    серийники, ImageUniqueID); не гарантирует проход модерации FB."""

    preset: str = "strong"  # "fb" | "strong"
    preserve_visual: bool = False
    synthetic_exif: bool = True


@dataclass(frozen=True)
class DeepAugmentOptions:
    """Пайплайн глубокой аугментации для обучения.

    Важно: pHash — робастный перцептивный хэш, он специально устойчив к мелким правкам.
    Поэтому опция `phash_breaker_passes` даёт несколько разных «микро-комбинаций» шагов,
    чтобы повысить шанс сильного сдвига pHash, оставаясь в заданных пределах.
    """

    # Изображения
    strip_metadata: bool = True
    noise_ratio: float = 0.005  # 0.5% (σ≈255*0.005)
    gaussian_blur_sigma: tuple[float, float] = (0.15, 0.45)  # «лёгкий» blur
    brightness_jitter: float = 0.05  # ±5%
    contrast_jitter: float = 0.05  # ±5%
    random_crop_frac: float = 0.02  # 2% от меньшей стороны, с ресайзом обратно
    micro_rotate_deg: float = 0.5  # до 0.5°
    resize_jitter: float = 0.01  # ±1%
    invisible_pattern_alpha: float = 0.01  # 1% прозрачность
    # Цвет / ICC (для вариативности бинарного потока и цветопредставления)
    # Передай пути к ICC-файлам (например: DisplayP3.icc, AdobeRGB1998.icc). sRGB доступен встроенно.
    icc_profile_paths: tuple[Path, ...] = ()
    allow_srgb_profile: bool = True
    # Совместимость: если задано — принудительно конвертировать/сохранять в AdobeRGB через ICC.
    adobe_rgb_icc_path: Path | None = None  # ICC AdobeRGB1998.icc
    # Усиление pHash-сдвига
    phash_breaker_passes: int = 3  # повторить комбинацию микро-правок несколько раз
    phash_pattern_freq: tuple[float, float] = (14.0, 32.0)  # частоты паттерна (в пикселях)
    # Итеративный контроль pHash (только изображения). similarity=1.0 означает идентичный pHash.
    target_phash_similarity: float | None = 0.85
    max_phash_iterations: int = 18

    # Видео (FFmpeg)
    video_blank_tail_seconds: tuple[float, float] = (0.15, 0.45)  # «пустые» кадры (чёрный хвост)
    video_flicker_hz: tuple[float, float] = (0.8, 2.4)  # мерцание (циклично)
    video_flicker_strength: tuple[float, float] = (0.008, 0.02)  # амплитуда bright


def strip_all_metadata_image(im: Image.Image) -> Image.Image:
    """Полная очистка метаданных на уровне Pillow: убираем EXIF/XMP/ICC, и сбрасываем im.info."""
    # Пиксели — отдельно; info — выбрасываем.
    im = ImageOps.exif_transpose(im)
    im = _flatten_to_rgb(im)
    return _pixels_without_sidecar_metadata(im)


def strip_all_metadata_file(input_path: Path, output_path: Path) -> None:
    """Полная очистка метаданных файла.

    - Для изображений: перекодирование в JPEG без EXIF/XMP/ICC.
    - Для видео: ffmpeg -map_metadata -1 -map_chapters -1.
    """
    suffix = input_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in IMAGE_INPUT_SUFFIXES:
        with Image.open(input_path) as im:
            if getattr(im, "is_animated", False):
                im.seek(0)
            im.load()
            # Гарантированная пересборка: frombytes (сброс im.info) + JPEG roundtrip в памяти,
            # чтобы уничтожить исходные сегменты/структуру заголовков.
            clean0 = strip_all_metadata_image(im)
            buf = io.BytesIO()
            clean0.save(
                buf,
                format="JPEG",
                quality=95,
                optimize=True,
                progressive=True,
                subsampling=2,
                **_jpeg_save_strip_metadata(),
            )
            buf.seek(0)
            with Image.open(buf) as j2:
                j2.load()
                clean1 = strip_all_metadata_image(j2)
            clean1.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
                progressive=True,
                subsampling=2,
                **_jpeg_save_strip_metadata(),
            )
        return
    if suffix in VIDEO_INPUT_SUFFIXES:
        ffmpeg = _require_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+bitexact",
            "-i",
            str(input_path),
        ] + _ffmpeg_sterile_mp4_metadata_args() + [
            "-dn",
            "-sn",
            "-c",
            "copy",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or "")[-2500:]
            raise RuntimeError(f"ffmpeg ошибка при очистке метаданных:\n{err}")
        return
    raise ValueError(f"Неподдерживаемый формат: {suffix}")


def add_gaussian_noise_and_blur(im: Image.Image, rng: random.Random, noise_ratio: float, blur_sigma: tuple[float, float]) -> Image.Image:
    """Шум (0.5%) + лёгкий Gaussian blur."""
    im = im.convert("RGB")
    sigma = max(0.0, float(noise_ratio)) * 255.0
    if sigma > 0 and np is not None:
        arr = np.asarray(im).astype(np.float32)
        noise = rng.normalvariate(0.0, 1.0)  # прогрев RNG, чтобы «соль» менялась
        _ = noise
        n = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + n, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB")
    elif sigma > 0:
        # fallback без numpy: очень лёгкий слой шума через effect_noise + alpha
        alpha = max(0.001, min(0.03, noise_ratio * 2.0))
        im = _full_frame_transparent_noise(im, rng, opacity=alpha)

    lo, hi = blur_sigma
    r = rng.uniform(float(lo), float(hi))
    if r > 1e-6:
        im = im.filter(ImageFilter.GaussianBlur(radius=r))
    return im


def randomize_brightness_contrast_and_crop(im: Image.Image, rng: random.Random, brightness_j: float, contrast_j: float, crop_frac: float) -> Image.Image:
    im = im.convert("RGB")
    b = 1.0 + rng.uniform(-abs(brightness_j), abs(brightness_j))
    c = 1.0 + rng.uniform(-abs(contrast_j), abs(contrast_j))
    im = ImageEnhance.Brightness(im).enhance(b)
    im = ImageEnhance.Contrast(im).enhance(c)
    if crop_frac > 0:
        w, h = im.size
        m = min(w, h)
        pad = int(round(m * float(crop_frac)))
        if pad > 0 and w - 2 * pad >= 2 and h - 2 * pad >= 2:
            left = rng.randint(0, pad)
            top = rng.randint(0, pad)
            right = rng.randint(0, pad)
            bottom = rng.randint(0, pad)
            cropped = im.crop((left, top, w - right, h - bottom))
            im = cropped.resize((w, h), resample=_pick_resample(rng))
    return im


def micro_rotate_and_resize(im: Image.Image, rng: random.Random, max_deg: float, resize_jitter: float) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    deg = rng.uniform(-abs(max_deg), abs(max_deg))
    if abs(deg) > 1e-6:
        im = im.rotate(deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    j = abs(resize_jitter)
    if j > 0:
        s = 1.0 + rng.uniform(-j, j)
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        if (nw, nh) != (w, h):
            im2 = im.resize((nw, nh), resample=_pick_resample(rng))
            if nw >= w and nh >= h:
                left = (nw - w) // 2
                top = (nh - h) // 2
                im = im2.crop((left, top, left + w, top + h))
            else:
                im = ImageOps.pad(im2, (w, h), method=_pick_resample(rng), color=(0, 0, 0), centering=(0.5, 0.5))
    return im


def overlay_invisible_pattern(im: Image.Image, rng: random.Random, alpha: float, freq_px: tuple[float, float]) -> Image.Image:
    """Невидимый паттерн (watermark) с прозрачностью ~1%.

    Делается как низкоамплитудная синусоидальная «решётка» + псевдослучайная фаза.
    Это сильнее влияет на частотные признаки (и pHash), чем одиночный пиксель.
    """
    im = im.convert("RGB")
    a = max(0.0, min(0.05, float(alpha)))
    if a <= 0:
        return im
    w, h = im.size
    if w < 2 or h < 2:
        return im
    if np is None:
        # fallback: используем уже существующий микро-слой шума с указанной альфой
        return _full_frame_transparent_noise(im, rng, opacity=a)

    fx = rng.uniform(float(freq_px[0]), float(freq_px[1]))
    fy = rng.uniform(float(freq_px[0]), float(freq_px[1]))
    phx = rng.uniform(0.0, 6.283185307179586)
    phy = rng.uniform(0.0, 6.283185307179586)
    yy, xx = np.mgrid[0:h, 0:w]
    # Паттерн [-1..1]
    patt = np.sin((xx / fx) * (2 * np.pi) + phx) * np.sin((yy / fy) * (2 * np.pi) + phy)
    # Сдвиг яркости на ±k (k маленькое)
    k = rng.uniform(0.75, 2.25)
    delta = (patt * k).astype(np.float32)
    arr = np.asarray(im).astype(np.float32)
    # «Прозрачность» реализуем как микс: base + a*delta
    arr = np.clip(arr + (a * 6.0) * delta[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def convert_to_adobe_rgb(im: Image.Image, adobe_icc_path: Path) -> Image.Image:
    """Конвертация цветового профиля в AdobeRGB (требуется ICC AdobeRGB1998.icc)."""
    if not adobe_icc_path or not adobe_icc_path.exists():
        raise RuntimeError(
            "Не найден ICC-профиль AdobeRGB. Передай путь в DeepAugmentOptions.adobe_rgb_icc_path "
            "(например, AdobeRGB1998.icc)."
        )
    im = im.convert("RGB")
    src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    dst = ImageCms.ImageCmsProfile(str(adobe_icc_path))
    return ImageCms.profileToProfile(im, src, dst, outputMode="RGB")


@lru_cache(maxsize=16)
def _icc_profile_bytes_from_path(p: str) -> bytes:
    return Path(p).read_bytes()


def _pick_output_icc_profile_bytes(rng: random.Random, opts: DeepAugmentOptions) -> tuple[bytes | None, Path | None]:
    """Возвращает (icc_bytes, icc_path_used) для embedding; sRGB — встроенный профиль."""
    candidates: list[Path | str] = []
    if opts.allow_srgb_profile:
        candidates.append("SRGB_BUILTIN")
    candidates.extend([p for p in opts.icc_profile_paths if p and p.exists()])
    if not candidates:
        return None, None
    chosen = rng.choice(candidates)
    if chosen == "SRGB_BUILTIN":
        return _srgb_iec61966_icc_profile_bytes(), None
    b = _icc_profile_bytes_from_path(str(chosen))
    return b, Path(chosen)


def _phash_64(im: Image.Image) -> int:
    """Перцептивный pHash 64-bit (DCT 32x32 -> 8x8 без DC). Требует numpy."""
    if np is None:
        raise RuntimeError("Для контроля pHash нужен numpy (pip install numpy).")
    g = im.convert("L").resize((32, 32), resample=Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    if cv2 is not None:
        dct = cv2.dct(a)
    else:
        n = 32
        k = np.arange(n, dtype=np.float32)[:, None]
        x = np.arange(n, dtype=np.float32)[None, :]
        mat = np.cos((np.pi / n) * (x + 0.5) * k)
        mat[0, :] *= 1.0 / np.sqrt(2.0)
        mat *= np.sqrt(2.0 / n)
        dct = mat @ a @ mat.T
    block = dct[:8, :8].copy()
    block[0, 0] = 0.0
    med = float(np.median(block))
    bits = (block > med).astype(np.uint8).reshape(-1)
    out = 0
    for i, b in enumerate(bits.tolist()):
        out |= (int(b) & 1) << i
    return out


def _phash_similarity(h1: int, h2: int) -> float:
    """1.0 = идентичный pHash; 0.0 = максимально разный (по Хэммингу на 64 битах)."""
    x = (h1 ^ h2) & ((1 << 64) - 1)
    dist = int(x).bit_count()
    return 1.0 - (dist / 64.0)


def deep_augment_image(input_path: Path, output_path: Path, opts: DeepAugmentOptions) -> None:
    """Глубокая аугментация изображения (PIL + optional numpy)."""
    raw = input_path.read_bytes()
    rng = _rng_for_bytes(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as im0:
        if getattr(im0, "is_animated", False):
            im0.seek(0)
        im0.load()
        base = strip_all_metadata_image(im0) if opts.strip_metadata else _flatten_to_rgb(ImageOps.exif_transpose(im0))

        target_sim = opts.target_phash_similarity
        if target_sim is not None:
            target_sim = float(target_sim)
            target_sim = max(0.0, min(1.0, target_sim))
        base_h = _phash_64(base) if target_sim is not None else 0

        im = base
        passes = max(1, int(opts.phash_breaker_passes))
        max_it = max(1, int(opts.max_phash_iterations))
        best_im = im
        best_sim = 1.0

        for _ in range(max_it):
            cand = im
            for _p in range(passes):
                cand = add_gaussian_noise_and_blur(cand, rng, opts.noise_ratio, opts.gaussian_blur_sigma)
                cand = randomize_brightness_contrast_and_crop(
                    cand, rng, opts.brightness_jitter, opts.contrast_jitter, opts.random_crop_frac
                )
                cand = micro_rotate_and_resize(cand, rng, opts.micro_rotate_deg, opts.resize_jitter)
                cand = overlay_invisible_pattern(cand, rng, opts.invisible_pattern_alpha, opts.phash_pattern_freq)
                cand = _nudge_one_pixel_for_unique_bytes(cand, rng)

            if target_sim is None:
                best_im = cand
                break

            sim = _phash_similarity(base_h, _phash_64(cand))
            if sim < best_sim:
                best_sim = sim
                best_im = cand
            if sim <= target_sim:
                best_im = cand
                best_sim = sim
                break

            im = cand

        im = best_im

        # ICC вариативность: случайно выбираем профиль и (по возможности) конвертируем sRGB->target.
        icc_bytes, icc_path_used = _pick_output_icc_profile_bytes(rng, opts)
        if opts.adobe_rgb_icc_path is not None:
            im = convert_to_adobe_rgb(im, opts.adobe_rgb_icc_path)
            icc_bytes = _icc_profile_bytes_from_path(str(opts.adobe_rgb_icc_path))
            icc_path_used = opts.adobe_rgb_icc_path
        elif icc_path_used is not None:
            try:
                src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
                dst = ImageCms.ImageCmsProfile(str(icc_path_used))
                im = ImageCms.profileToProfile(im.convert("RGB"), src, dst, outputMode="RGB")
            except Exception:
                im = im.convert("RGB")

        # Финальное сохранение: без метаданных (по умолчанию) + рандомизация JPEG-параметров + случайный ICC
        q = rng.randint(86, 96)
        subs = rng.choice([2, 2, 1, 0])
        prog = rng.random() < 0.75
        save_kwargs = _jpeg_save_strip_metadata() if opts.strip_metadata else {}
        if icc_bytes is not None:
            save_kwargs = dict(save_kwargs)
            save_kwargs["icc_profile"] = icc_bytes
        im.save(
            output_path,
            format="JPEG",
            quality=q,
            optimize=True,
            progressive=prog,
            subsampling=subs,
            **save_kwargs,
        )


def deep_augment_video(input_path: Path, output_path: Path, opts: DeepAugmentOptions) -> None:
    """Глубокая аугментация видео через FFmpeg:
    - очистка метаданных
    - добавление чёрного хвоста (пустые кадры)
    - цикличное мерцание яркости
    """
    ffmpeg = _require_ffmpeg()
    ffprobe = _require_ffprobe()
    rng = _rng_for_bytes(input_path.read_bytes())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = _probe_video_size(input_path, ffprobe, ffmpeg)
    if w <= 0 or h <= 0:
        w, h = 1280, 720
    # для кодеков лучше чётные размеры
    w2 = max(2, (w // 2) * 2)
    h2 = max(2, (h // 2) * 2)

    tail = rng.uniform(float(opts.video_blank_tail_seconds[0]), float(opts.video_blank_tail_seconds[1]))
    tail = max(0.0, min(2.0, tail))
    hz = rng.uniform(float(opts.video_flicker_hz[0]), float(opts.video_flicker_hz[1]))
    amp = rng.uniform(float(opts.video_flicker_strength[0]), float(opts.video_flicker_strength[1]))

    has_audio = _has_audio_stream(input_path, ffprobe)

    # Мерцание: eq brightness синусом от времени.
    # Хвост: concat с color=black нужной длительности.
    # В конце форматируем в yuv420p для совместимости.
    flicker = f"eq=brightness='{amp:.6f}*sin(2*PI*{hz:.6f}*t)'"
    if tail > 1e-6:
        fc = (
            f"[0:v]scale={w2}:{h2},{flicker},format=yuv420p[v0];"
            f"color=c=black:s={w2}x{h2}:d={tail:.6f},format=yuv420p[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[v]"
        )
        map_v = ["-filter_complex", fc, "-map", "[v]"]
    else:
        vf = f"scale={w2}:{h2},{flicker},format=yuv420p"
        map_v = ["-vf", vf]

    venc = ["-c:v", "libx264", "-crf", str(rng.randint(20, 25)), "-preset", rng.choice(["veryfast", "faster", "fast"]), "-pix_fmt", "yuv420p"]

    if has_audio:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
        ] + _ffmpeg_sterile_mp4_metadata_args() + [
            "-dn",
            "-sn",
        ] + map_v + [
            "-map",
            "0:a?",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ] + venc + [str(output_path)]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
        ] + _ffmpeg_sterile_mp4_metadata_args() + [
            "-dn",
            "-sn",
        ] + map_v + ["-an"] + venc + [str(output_path)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "")[-2500:]
        raise RuntimeError(f"ffmpeg ошибка при deep-augmentation:\n{err}")


def deep_augment_file(input_path: Path, output_path: Path, opts: DeepAugmentOptions) -> None:
    """Единая точка входа для deep-augmentation (картинки/видео)."""
    suffix = input_path.suffix.lower()
    if suffix in IMAGE_INPUT_SUFFIXES:
        deep_augment_image(input_path, output_path, opts)
        return
    if suffix in VIDEO_INPUT_SUFFIXES:
        deep_augment_video(input_path, output_path, opts)
        return
    raise ValueError(f"Неподдерживаемый формат: {suffix}")


def _is_strong(options: UniqulizeOptions) -> bool:
    return (options.preset or "fb").lower().strip() == "strong"


def _rng_for_bytes(data: bytes) -> random.Random:
    """Сид RNG: хэш файла + криптосоль на каждый вызов — два пользователя с одним и тем же крео получают разную уникализацию."""
    material = hashlib.sha256(data + secrets.token_bytes(32)).digest()
    seed = int.from_bytes(material[:8], "big", signed=False)
    return random.Random(seed)


def _image_params(strong: bool, preserve_visual: bool = True) -> dict:
    if not strong and preserve_visual:
        # fb + сохранение кадра: без масштаба/обрезки/поворота/цветовых фильтров; только новый JPEG-битстрим без EXIF/ICC
        return {
            "identity_color": True,
            "resize_pct": (0.0, 0.0),
            "rotate_deg": (0.0, 0.0),
            "edge_crop": (0.0, 0.0),
            "flip_prob": 0.0,
            "brightness": (1.0, 1.0),
            "contrast": (1.0, 1.0),
            "color": (1.0, 1.0),
            "sharpness": (1.0, 1.0),
            "gamma": (1.0, 1.0),
            "blur_r": (0.0, 0.0),
            "unsharp": ((1.0, 1.0), (0, 0), (0, 0)),
            "noise_patches": (0, 0),
            "noise_amp": (0, 0),
            "jpeg_q": (96, 99),
            "micro_noise_amp": (0, 0),
            "post_roundtrip_nudge": (1.0, 1.0),
        }
    if not strong:
        # fb без preserve_visual: сильнее ломаем след по файлу (опционально)
        return {
            "identity_color": False,
            "resize_pct": (0.004, 0.022),
            "rotate_deg": (-0.72, 0.72),
            "edge_crop": (0.006, 0.022),
            "brightness": (0.988, 1.018),
            "contrast": (0.988, 1.022),
            "color": (0.988, 1.024),
            "sharpness": (0.96, 1.045),
            "gamma": (0.965, 1.038),
            "blur_r": (0.18, 0.62),
            "unsharp": ((1.0, 2.4), (105, 195), (1, 5)),
            "noise_patches": (14, 38),
            "noise_amp": (2, 4),
            "flip_prob": 0.12,
            "jpeg_q": (86, 95),
            "micro_noise_amp": (1, 3),
            "post_roundtrip_nudge": (0.996, 1.004),
        }
    return {
        "identity_color": False,
        "resize_pct": (0.006, 0.028),
        "rotate_deg": (-1.4, 1.4),
        "edge_crop": (0.008, 0.028),
        "brightness": (0.985, 1.028),
        "contrast": (0.982, 1.04),
        "color": (0.985, 1.045),
        "sharpness": (0.95, 1.08),
        "gamma": (0.94, 1.06),
        "blur_r": (0.25, 0.85),
        "unsharp": ((1.2, 2.8), (120, 220), (1, 6)),
        "noise_patches": (18, 55),
        "noise_amp": (2, 7),
        "flip_prob": 0.2,
        "jpeg_q": (84, 94),
        "micro_noise_amp": (2, 5),
        "post_roundtrip_nudge": (0.992, 1.008),
    }


def _hsv_nudge_rgb(im: Image.Image, rng: random.Random, strong: bool) -> Image.Image:
    im = im.convert("RGB")
    hsv = im.convert("HSV")
    h, s, v = hsv.split()
    dh = rng.randint(-4, 4) if strong else rng.randint(-2, 2)

    def nudge_h(x: int) -> int:
        return (x + dh) % 256

    h = h.point(nudge_h)
    sm = rng.uniform(0.97, 1.04) if strong else rng.uniform(0.985, 1.025)
    vm = rng.uniform(0.97, 1.04) if strong else rng.uniform(0.988, 1.02)

    s = s.point(lambda x: max(0, min(255, int(x * sm))))
    v = v.point(lambda x: max(0, min(255, int(x * vm))))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _gamma_rgb(im: Image.Image, rng: random.Random, g_lo: float, g_hi: float) -> Image.Image:
    g = rng.uniform(g_lo, g_hi)
    inv = 1.0 / max(g, 1e-6)

    def curve(p: int) -> int:
        x = p / 255.0
        y = x**inv
        return max(0, min(255, int(y * 255 + 0.5)))

    return im.point(curve) if im.mode == "RGB" else im.convert("RGB").point(curve)


def _edge_crop_to_original(im: Image.Image, rng: random.Random, edge_frac: tuple[float, float]) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    lo, hi = edge_frac
    m = min(w, h)
    t = rng.uniform(lo, hi) * m
    # remove roughly t pixels total from perimeter (split random)
    left = int(rng.uniform(0, t * 0.5))
    right = int(rng.uniform(0, t * 0.5))
    top = int(rng.uniform(0, t * 0.5))
    bottom = int(rng.uniform(0, t * 0.5))
    left = min(left, w // 4)
    right = min(right, w // 4)
    top = min(top, h // 4)
    bottom = min(bottom, h // 4)
    if left + right >= w - 2 or top + bottom >= h - 2:
        return im
    cropped = im.crop((left, top, w - right, h - bottom))
    return cropped.resize((w, h), resample=_pick_resample(rng))


def _noise_patches(im: Image.Image, rng: random.Random, n_lo: int, n_hi: int, strength: int) -> Image.Image:
    if strength <= 0:
        return im
    im = im.convert("RGB")
    px = im.copy()
    load = px.load()
    w, h = px.size
    n = rng.randint(n_lo, n_hi)
    for _ in range(n):
        pw = rng.randint(max(4, w // 100), max(8, w // 12))
        ph = rng.randint(max(4, h // 100), max(8, h // 12))
        x0 = rng.randint(0, max(0, w - pw))
        y0 = rng.randint(0, max(0, h - ph))
        ch = rng.choice((0, 1, 2))
        for yy in range(y0, min(h, y0 + ph)):
            for xx in range(x0, min(w, x0 + pw)):
                r, g, b = load[xx, yy]
                nval = rng.randint(-strength, strength)
                if ch == 0:
                    r = max(0, min(255, r + nval))
                elif ch == 1:
                    g = max(0, min(255, g + nval))
                else:
                    b = max(0, min(255, b + nval))
                load[xx, yy] = (r, g, b)
    return px


def _flatten_to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    if im.mode == "LA":
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    if im.mode == "P":
        if im.info.get("transparency") is not None:
            return _flatten_to_rgb(im.convert("RGBA"))
        return im.convert("RGB")
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _micro_luminance_noise(im: Image.Image, rng: random.Random, amp_lo: int, amp_hi: int) -> Image.Image:
    """Редкий шум по пикселям — ломает побитовые совпадения без заметной деградации."""
    if amp_hi <= 0:
        return im.convert("RGB")
    im = im.convert("RGB")
    px = im.copy()
    load = px.load()
    w, h = px.size
    amp = max(1, rng.randint(amp_lo, amp_hi))
    touches = max(400, (w * h) // 3500)
    for _ in range(touches):
        xx = rng.randint(0, w - 1)
        yy = rng.randint(0, h - 1)
        r, g, b = load[xx, yy]
        n = rng.randint(-amp, amp)
        load[xx, yy] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return px


def _pick_resample(rng: random.Random) -> Image.Resampling:
    return rng.choice(
        (
            Image.Resampling.LANCZOS,
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
        )
    )


def _full_frame_transparent_noise(im: Image.Image, rng: random.Random, opacity: float | None = None) -> Image.Image:
    """Микро-слой шума на весь кадр (альфа ~0.005) — меняет битстрим без видимых артефактов."""
    op = opacity if opacity is not None else rng.uniform(0.004, 0.006)
    op = max(0.001, min(0.02, float(op)))
    im = im.convert("RGB")
    w, h = im.size
    if w < 1 or h < 1:
        return im
    sigma = rng.uniform(0.75, 2.1)
    noise_l = Image.effect_noise((w, h), sigma).convert("L")
    alpha_v = max(1, min(255, int(round(255 * op))))
    a = Image.new("L", (w, h), alpha_v)
    noise_rgba = Image.merge("RGBA", (noise_l, noise_l, noise_l, a))
    base = im.convert("RGBA")
    return Image.alpha_composite(base, noise_rgba).convert("RGB")


def _jpeg_save_strip_metadata() -> dict:
    """Полный сброс EXIF/XMP/ICC при записи JPEG (в т.ч. цветовые профили, без переноса IPTC/XMP из редактора)."""
    return {"exif": b"", "icc_profile": None, "xmp": b""}


def _jpeg_save_strip_metadata_final(*, synthetic_exif: bool) -> dict:
    """Финальный JPEG: без чужого EXIF; при synthetic_exif — стандартный sRGB ICC (IEC61966-2.1)."""
    if synthetic_exif:
        return {"exif": b"", "icc_profile": _srgb_iec61966_icc_profile_bytes(), "xmp": b""}
    return {"exif": b"", "icc_profile": None, "xmp": b""}


def _pixels_without_sidecar_metadata(im: Image.Image) -> Image.Image:
    """Убирает привязку к im.info (XMP/IPTC и пр.), пересобирая RGB из буфера — новый кадр без скрытых тегов PIL."""
    im = im.convert("RGB")
    return Image.frombytes("RGB", im.size, im.tobytes())


def _nudge_one_pixel_for_unique_bytes(im: Image.Image, rng: random.Random) -> Image.Image:
    """Один пиксель ±1 по каналу — незаметно глазу, гарантированно меняет битстрим/MD5 вместе с quality."""
    im = im.convert("RGB")
    px = im.load()
    w, h = im.size
    if w < 1 or h < 1:
        return im
    x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
    r, g, b = px[x, y]
    ch = rng.choice((0, 1, 2))
    delta = rng.choice((-1, 1))
    if ch == 0:
        r = max(0, min(255, r + delta))
    elif ch == 1:
        g = max(0, min(255, g + delta))
    else:
        b = max(0, min(255, b + delta))
    px[x, y] = (r, g, b)
    return im


# Только Apple iPhone и Samsung Galaxy — Make/Model/объектив/строка ПО (~50 профилей)
_SYNTHETIC_DEVICE_PROFILES: tuple[tuple[bytes, bytes, bytes, bytes], ...] = (
    (b"Apple", b"iPhone 11", b"iPhone 11 back dual camera", b"15.7.1"),
    (b"Apple", b"iPhone 11 Pro", b"iPhone 11 Pro back triple camera", b"15.7.1"),
    (b"Apple", b"iPhone 11 Pro Max", b"iPhone 11 Pro Max back triple camera", b"15.7.1"),
    (b"Apple", b"iPhone 12 mini", b"iPhone 12 mini back dual camera", b"16.6.1"),
    (b"Apple", b"iPhone 12", b"iPhone 12 back dual camera", b"16.6.1"),
    (b"Apple", b"iPhone 12 Pro", b"iPhone 12 Pro back triple camera", b"16.6.1"),
    (b"Apple", b"iPhone 12 Pro Max", b"iPhone 12 Pro Max back triple camera", b"16.6.1"),
    (b"Apple", b"iPhone 13 mini", b"iPhone 13 mini back dual wide camera", b"16.5.1"),
    (b"Apple", b"iPhone 13", b"iPhone 13 back dual wide camera", b"16.5.1"),
    (b"Apple", b"iPhone 13 Pro", b"iPhone 13 Pro back triple camera", b"16.5.1"),
    (b"Apple", b"iPhone 13 Pro Max", b"iPhone 13 Pro Max back triple camera", b"16.5.1"),
    (b"Apple", b"iPhone SE (3rd generation)", b"iPhone SE (3rd generation) back camera", b"16.5.1"),
    (b"Apple", b"iPhone 14", b"iPhone 14 back dual camera", b"17.3.1"),
    (b"Apple", b"iPhone 14 Plus", b"iPhone 14 Plus back dual camera", b"17.3.1"),
    (b"Apple", b"iPhone 14 Pro", b"iPhone 14 Pro back triple camera", b"17.3.1"),
    (b"Apple", b"iPhone 14 Pro Max", b"iPhone 14 Pro Max back triple camera", b"17.3.1"),
    (b"Apple", b"iPhone 15", b"iPhone 15 back dual camera", b"17.4.1"),
    (b"Apple", b"iPhone 15 Plus", b"iPhone 15 Plus back dual camera", b"17.4.1"),
    (b"Apple", b"iPhone 15 Pro", b"iPhone 15 Pro back triple camera", b"17.4.1"),
    (b"Apple", b"iPhone 15 Pro Max", b"iPhone 15 Pro Max back triple camera", b"17.4.1"),
    (b"Apple", b"iPhone 16", b"iPhone 16 back dual camera", b"18.0.1"),
    (b"Apple", b"iPhone 16 Plus", b"iPhone 16 Plus back dual camera", b"18.0.1"),
    (b"Apple", b"iPhone 16 Pro", b"iPhone 16 Pro back triple camera", b"18.0.1"),
    (b"Apple", b"iPhone 16 Pro Max", b"iPhone 16 Pro Max back triple camera", b"18.0.1"),
    (b"Samsung", b"SM-G991B", b"Galaxy S21 main camera", b"G991BXXU9EWH1"),
    (b"Samsung", b"SM-G996B", b"Galaxy S21+ main camera", b"G996BXXU9EWH1"),
    (b"Samsung", b"SM-G998B", b"Galaxy S21 Ultra main", b"G998BXXU9EUL5"),
    (b"Samsung", b"SM-G990B", b"Galaxy S21 FE main", b"G990BXXU2EWH1"),
    (b"Samsung", b"SM-S901B", b"Galaxy S22 main", b"S901BXXU7CWDA"),
    (b"Samsung", b"SM-S906B", b"Galaxy S22+ main", b"S906BXXU7CWDA"),
    (b"Samsung", b"SM-S908B", b"Galaxy S22 Ultra main", b"S908BXXU6EWH1"),
    (b"Samsung", b"SM-S911B", b"Galaxy S23 main", b"S911BXXU3CWAI"),
    (b"Samsung", b"SM-S916B", b"Galaxy S23+ main", b"S916BXXU3CWAI"),
    (b"Samsung", b"SM-S918B", b"Galaxy S23 Ultra main", b"S918BXXU3CWAI"),
    (b"Samsung", b"SM-S921B", b"Galaxy S24 main", b"S921BXXU1AXB5"),
    (b"Samsung", b"SM-S926B", b"Galaxy S24+ main", b"S926BXXU1AXB5"),
    (b"Samsung", b"SM-S928B", b"Galaxy S24 Ultra main", b"S928BXXU1AXB5"),
    (b"Samsung", b"SM-S721B", b"Galaxy S24 FE main", b"S721BXXU1AXB5"),
    (b"Samsung", b"SM-A546B", b"Galaxy A54 main", b"A546BXXU6CWC2"),
    (b"Samsung", b"SM-A556B", b"Galaxy A55 main", b"A556BXXU2AXD1"),
    (b"Samsung", b"SM-A536B", b"Galaxy A53 main", b"A536BXXU6CWC2"),
    (b"Samsung", b"SM-A346B", b"Galaxy A34 main", b"A346BXXU2CWC2"),
    (b"Samsung", b"SM-A336B", b"Galaxy A33 main", b"A336BXXU4CWC2"),
    (b"Samsung", b"SM-A528B", b"Galaxy A52s main", b"A528BXXU2EWH1"),
    (b"Samsung", b"SM-F731B", b"Galaxy Z Flip5 main", b"F731BXXU1AWB"),
    (b"Samsung", b"SM-F946B", b"Galaxy Z Fold5 main", b"F946BXXU1AWB"),
    (b"Samsung", b"SM-F741B", b"Galaxy Z Flip6 main", b"F741BXXU1AXB"),
    (b"Samsung", b"SM-F956B", b"Galaxy Z Fold6 main", b"F956BXXU1AXB"),
    (b"Samsung", b"SM-N981B", b"Galaxy Note20 main", b"N981BXXU6FUB1"),
    (b"Samsung", b"SM-N986B", b"Galaxy Note20 Ultra main", b"N986BXXU6FUB1"),
)


def _random_exif_datetime_recent_hours(rng: random.Random) -> bytes:
    """DateTime / DateTimeOriginal / DateTimeDigitized — в пределах последних нескольких часов."""
    hours = rng.uniform(0.25, 8.0)
    end = datetime.now()
    t = end - timedelta(seconds=rng.uniform(0.0, hours * 3600.0))
    return t.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")


def _random_ascii_serial(rng: random.Random, length: int) -> bytes:
    alphabet = b"ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return bytes(rng.choice(alphabet) for _ in range(length))


def _make_exif_thumbnail_jpeg(path: Path, rng: random.Random) -> bytes | None:
    """Миниатюра для EXIF IFD (как у «мобильного» снимка)."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            mx = rng.randint(140, 220)
            im.thumbnail((mx, mx), resample=_pick_resample(rng))
            buf = io.BytesIO()
            im.save(
                buf,
                format="JPEG",
                quality=rng.randint(72, 88),
                optimize=True,
                subsampling=2,
                **_jpeg_save_strip_metadata(),
            )
            return buf.getvalue()
    except Exception:
        return None


def _inject_synthetic_exif_into_jpeg_file(path: Path, rng: random.Random) -> None:
    """После чистого JPEG: piexif — Make/Model (iPhone/Samsung), недавние даты, серийники, ImageUniqueID; без GPS; thumbnail в EXIF."""
    make, model, lens_model, software = rng.choice(_SYNTHETIC_DEVICE_PROFILES)
    dt = _random_exif_datetime_recent_hours(rng)
    sub = f"{rng.randint(0, 999):03d}".encode("ascii")
    img_uid = secrets.token_hex(16).upper().encode("ascii")
    body = _random_ascii_serial(rng, rng.randint(10, 14))
    lens_sn = _random_ascii_serial(rng, rng.randint(9, 12))

    zeroth_ifd = {
        piexif.ImageIFD.Make: make,
        piexif.ImageIFD.Model: model,
        piexif.ImageIFD.Software: software,
        piexif.ImageIFD.DateTime: dt,
        piexif.ImageIFD.Orientation: 1,
        piexif.ImageIFD.XResolution: (72, 1),
        piexif.ImageIFD.YResolution: (72, 1),
        piexif.ImageIFD.ResolutionUnit: 2,
    }
    exif_ifd = {
        piexif.ExifIFD.ExifVersion: b"0232",
        piexif.ExifIFD.DateTimeOriginal: dt,
        piexif.ExifIFD.DateTimeDigitized: dt,
        piexif.ExifIFD.SubSecTimeOriginal: sub,
        piexif.ExifIFD.SubSecTimeDigitized: sub,
        piexif.ExifIFD.SubSecTime: sub,
        piexif.ExifIFD.LensMake: make,
        piexif.ExifIFD.LensModel: lens_model,
        piexif.ExifIFD.BodySerialNumber: body,
        piexif.ExifIFD.LensSerialNumber: lens_sn,
        piexif.ExifIFD.ImageUniqueID: img_uid,
        piexif.ExifIFD.ColorSpace: 1,
        piexif.ExifIFD.WhiteBalance: rng.choice((0, 1)),
        piexif.ExifIFD.ExposureProgram: 2,
        piexif.ExifIFD.MeteringMode: rng.choice((1, 2, 5, 6)),
        piexif.ExifIFD.Flash: rng.choice((0, 16, 24)),
        piexif.ExifIFD.ISOSpeedRatings: rng.randint(40, 640),
        piexif.ExifIFD.FocalLength: (rng.randint(17, 80), 10),
        piexif.ExifIFD.FNumber: (rng.randint(17, 28), 10),
        piexif.ExifIFD.ExposureTime: (1, rng.randint(40, 400)),
        piexif.ExifIFD.FocalLengthIn35mmFilm: rng.randint(24, 52),
        piexif.ExifIFD.ExposureBiasValue: (rng.randint(-10, 10), 10),
    }
    thumb = _make_exif_thumbnail_jpeg(path, rng)
    exif_dict: dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": {}, "1st": {}}
    if thumb:
        exif_dict["thumbnail"] = thumb
    try:
        exif_bytes = piexif.dump(exif_dict)
        jpeg = path.read_bytes()
        out = io.BytesIO()
        piexif.insert(exif_bytes, jpeg, out)
        path.write_bytes(out.getvalue())
    except Exception:
        pass


def _jpeg_generation_roundtrip(im: Image.Image, rng: random.Random, p: dict) -> Image.Image:
    """JPEG в памяти: новые таблицы квантования и битстрим, без EXIF/ICC."""
    qmin, qmax = p["jpeg_q"]
    q1 = rng.randint(qmin, max(qmin + 1, qmax))
    if p.get("identity_color"):
        subs = rng.choice([0, 0, 1, 1, 2])
    else:
        subs = rng.choice([2, 2, 1])
    buf = io.BytesIO()
    prog_thr = 0.42 if p.get("identity_color") else 0.88
    im.save(
        buf,
        format="JPEG",
        quality=q1,
        optimize=True,
        progressive=rng.random() < prog_thr,
        subsampling=subs,
        **_jpeg_save_strip_metadata(),
    )
    buf.seek(0)
    with Image.open(buf) as j2:
        j2.load()
        out = j2.convert("RGB")
    lo, hi = p["post_roundtrip_nudge"]
    if hi - lo > 1e-12 and not (abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9):
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(lo, hi))
    return out


def uniqulize_image(input_path: Path, output_path: Path, options: UniqulizeOptions) -> None:
    """Уникализация картинки для выдачи в боте (поток FB).

    1) Снятие исходных метаданных: exif_transpose → RGB → `_pixels_without_sidecar_metadata` (без XMP/IPTC/im.info),
       перекодирование JPEG без EXIF/XMP/ICC (`_jpeg_save_strip_metadata`).
    2) Эмуляция устройства: `piexif`, только Make/Model iPhone/Samsung (`_SYNTHETIC_DEVICE_PROFILES`), `synthetic_exif`.
    3) DateTime / DateTimeOriginal / DateTimeDigitized — случайно в последних часах (`_random_exif_datetime_recent_hours`).
    4) Смена MD5: roundtrip JPEG с quality 96–99 (preserve), финальное сохранение с тем же диапазоном + `_nudge_one_pixel_for_unique_bytes`.
    5) Микро-шум на весь кадр; при synthetic_exif — sRGB ICC + EXIF с миниатюрой (piexif). Видео — `uniqulize_video`.
    """
    raw = input_path.read_bytes()
    rng = _rng_for_bytes(raw)
    strong = _is_strong(options)
    preserve_visual = bool(getattr(options, "preserve_visual", True)) and not strong
    p = _image_params(strong, preserve_visual)

    with Image.open(input_path) as im:
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
        im = ImageOps.exif_transpose(im)
        im = _flatten_to_rgb(im)
        im = _pixels_without_sidecar_metadata(im)

        im = _edge_crop_pad_pipeline(im, rng, p)

        if not p.get("identity_color"):
            deg = rng.uniform(*p["rotate_deg"])
            if abs(deg) > 1e-3:
                im = im.rotate(deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

            if rng.random() < p["flip_prob"]:
                im = ImageOps.mirror(im)

            im = _hsv_nudge_rgb(im, rng, strong)
            im = _gamma_rgb(im, rng, *p["gamma"])

            im = ImageEnhance.Brightness(im).enhance(rng.uniform(*p["brightness"]))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(*p["contrast"]))
            im = ImageEnhance.Color(im).enhance(rng.uniform(*p["color"]))
            im = ImageEnhance.Sharpness(im).enhance(rng.uniform(*p["sharpness"]))

            br_lo, br_hi = p["blur_r"]
            im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(br_lo, br_hi)))
            ur_lo, ur_hi = p["unsharp"][0]
            pc_lo, pc_hi = p["unsharp"][1]
            th_lo, th_hi = p["unsharp"][2]
            im = im.filter(
                ImageFilter.UnsharpMask(
                    radius=rng.uniform(ur_lo, ur_hi),
                    percent=int(rng.randint(pc_lo, pc_hi)),
                    threshold=int(rng.randint(th_lo, th_hi)),
                )
            )

            im = _noise_patches(
                im,
                rng,
                p["noise_patches"][0],
                p["noise_patches"][1],
                rng.randint(*p["noise_amp"]),
            )

            ma = p["micro_noise_amp"]
            im = _micro_luminance_noise(im, rng, ma[0], ma[1])

        im = _jpeg_generation_roundtrip(im, rng, p)

        qmin, qmax = p["jpeg_q"]
        quality = rng.randint(qmin, max(qmin + 1, qmax))
        if preserve_visual:
            subsampling = rng.choice([0, 0, 1, 1, 2])
            progressive = rng.random() < 0.5
        else:
            subsampling = rng.choice([2, 2, 1])
            progressive = rng.random() < 0.9

        im = _nudge_one_pixel_for_unique_bytes(im, rng)
        im = _full_frame_transparent_noise(im, rng)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        syn = bool(getattr(options, "synthetic_exif", True))
        im.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=progressive,
            subsampling=subsampling,
            **_jpeg_save_strip_metadata_final(synthetic_exif=syn),
        )
        if syn:
            _inject_synthetic_exif_into_jpeg_file(output_path, rng)


def _edge_crop_pad_pipeline(im: Image.Image, rng: random.Random, p: dict) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    resize_pct = rng.uniform(*p["resize_pct"])
    if resize_pct > 0:
        nw = max(1, int(w * (1.0 + (resize_pct if rng.random() < 0.5 else -resize_pct))))
        nh = max(1, int(h * (1.0 + (resize_pct if rng.random() < 0.5 else -resize_pct))))
        im = im.resize((nw, nh), resample=_pick_resample(rng))

        if nw >= w and nh >= h:
            left = (nw - w) // 2
            top = (nh - h) // 2
            im = im.crop((left, top, left + w, top + h))
        else:
            im = ImageOps.pad(im, (w, h), method=_pick_resample(rng), color=(0, 0, 0), centering=(0.5, 0.5))

    im = _edge_crop_to_original(im, rng, p["edge_crop"])
    return im


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    local = _ensure_local_ffmpeg()
    if local:
        return local
    raise RuntimeError(
        "ffmpeg не найден. Установи ffmpeg и добавь в PATH, либо задай путь через "
        "переменную окружения UNIQUILIZER_FFMPEG/FFMPEG_PATH."
    )


def _require_ffprobe() -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    _ = _ensure_local_ffmpeg()
    return shutil.which("ffprobe") or _local_ffprobe_path()


def _ffmpeg_sterile_mp4_metadata_args() -> list[str]:
    """Аргументы FFmpeg для максимально «стерильного» MP4:
    - вычистить глобальные и потоковые теги
    - убрать главы/таймкоды/доп. дорожки
    - минимизировать «следы» в контейнере
    """
    # Важно: часть тегов может быть как глобальной, так и потоковой.
    # Поэтому обнуляем и то, и то.
    global_tags = [
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-metadata",
        "encoder=",
        "-metadata",
        "encoded_by=",
        "-metadata",
        "creation_time=",
        "-metadata",
        "title=",
        "-metadata",
        "language=",
        "-metadata",
        "vendor_id=",
        "-metadata",
        "comment=",
    ]
    stream_tags = [
        "-metadata:s:v",
        "handler_name=",
        "-metadata:s:v",
        "language=",
        "-metadata:s:v",
        "vendor_id=",
        "-metadata:s:v",
        "comment=",
        "-metadata:s:v",
        "title=",
        "-metadata:s:v",
        "encoded_by=",
        "-metadata:s:v",
        "creation_time=",
        "-metadata:s:a",
        "handler_name=",
        "-metadata:s:a",
        "language=",
        "-metadata:s:a",
        "vendor_id=",
        "-metadata:s:a",
        "comment=",
        "-metadata:s:a",
        "title=",
        "-metadata:s:a",
        "encoded_by=",
        "-metadata:s:a",
        "creation_time=",
    ]
    # `-write_tmcd 0` нужен, чтобы не записывать таймкод-дорожку в MOV/MP4.
    container = [
        "-write_tmcd",
        "0",
        "-movflags",
        "+faststart",
        "-brand",
        "mp42",
    ]
    return global_tags + stream_tags + container


def _project_data_dir() -> Path:
    base = Path(__file__).resolve().parent
    p = base / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local_ffmpeg_root() -> Path:
    return _project_data_dir() / "ffmpeg"


def _local_ffmpeg_path() -> str | None:
    exe = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    p1 = _local_ffmpeg_root() / "bin" / exe
    if p1.exists():
        return str(p1)
    # fallback: если в архиве лежит вложенная папка ffmpeg-*/bin/ffmpeg.exe
    for cand in (_local_ffmpeg_root()).glob("**/bin/" + exe):
        if cand.exists():
            return str(cand)
    return None


def _local_ffprobe_path() -> str | None:
    exe = "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"
    p1 = _local_ffmpeg_root() / "bin" / exe
    if p1.exists():
        return str(p1)
    for cand in (_local_ffmpeg_root()).glob("**/bin/" + exe):
        if cand.exists():
            return str(cand)
    return None


def _ensure_local_ffmpeg() -> str | None:
    """Пытается найти ffmpeg без автоскачивания.

    На Windows автоскачивание часто блокируется политиками (WinError 4551).
    Поэтому тут только:
    - переменная окружения `UNIQUILIZER_FFMPEG` / `FFMPEG_PATH`
    - локальная папка проекта data/ffmpeg/bin
    - типовые места установки (chocolatey/winget/ручная распаковка)
    """
    env = (os.environ.get("UNIQUILIZER_FFMPEG") or os.environ.get("FFMPEG_PATH") or "").strip()
    if env:
        p = Path(env)
        if p.exists():
            bin_dir = str(p.parent)
            os_path = str(os.environ.get("PATH", ""))
            if bin_dir not in os_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os_path
            return str(p)

    # Локальный ffmpeg (data/ffmpeg) используем только если явно разрешили —
    # иначе на Windows его часто блокируют политики (WinError 4551).
    allow_local = (os.environ.get("UNIQUILIZER_ALLOW_LOCAL_FFMPEG") or "").strip() in ("1", "true", "yes", "on")
    if allow_local:
        local = _local_ffmpeg_path()
        if local:
            bin_dir = str(Path(local).parent)
            os_path = str(os.environ.get("PATH", ""))
            if bin_dir not in os_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os_path
            return local

    if not sys.platform.startswith("win"):
        return None

    exe = "ffmpeg.exe"
    common = [
        # Chocolatey
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
        Path(r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe"),
        # Частое ручное место
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        # Winget-пакеты иногда ставят в WindowsApps, но оттуда без прав часто не запустить; не полагаемся.
    ]
    for p in common:
        if p.exists():
            bin_dir = str(p.parent)
            os_path = str(os.environ.get("PATH", ""))
            if bin_dir not in os_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os_path
            return str(p)

    return None


def _has_audio_stream(input_path: Path, ffprobe: str | None) -> bool:
    if not ffprobe:
        return False
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def _probe_audio_channels(input_path: Path, ffprobe: str | None) -> int:
    if not ffprobe:
        return 2
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if r.returncode != 0:
        return 2
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return 2
    s = line[0].strip()
    if s.isdigit():
        return max(1, int(s))
    return 2


def _probe_video_fps(input_path: Path, ffprobe: str | None) -> float:
    """Пытаемся достать FPS через ffprobe (avg_frame_rate/r_frame_rate)."""
    if not ffprobe:
        return 0.0
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=nk=1:nw=1",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if r.returncode != 0:
        return 0.0
    lines = [(x or "").strip() for x in (r.stdout or "").splitlines() if (x or "").strip()]
    for s in lines:
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
        if m:
            num = float(m.group(1))
            den = float(m.group(2))
            if den > 0:
                fps = num / den
                if 0.1 < fps < 1000:
                    return fps
        # Иногда ffprobe может вернуть просто число
        try:
            fps2 = float(s)
            if 0.1 < fps2 < 1000:
                return fps2
        except ValueError:
            pass
    return 0.0


def _video_params(strong: bool) -> dict:
    if not strong:
        return {
            "crop_frac": (0.01, 0.03),
            "scale_jitter": (0.997, 1.003),
            "hue_rad": (-0.12, 0.12),
            "sat_mul": (0.97, 1.06),
            "brightness": (-0.035, 0.035),
            "contrast": (0.97, 1.045),
            "unsharp": ((5, 7), (0.35, 0.85)),
            "noise": (1, 5),
            "crf": (21, 24),
            "preset": ("veryfast", "faster", "fast"),
            "gop": (60, 150),
            "af_vol": (-1.5, 1.5),
            "atempo": (0.998, 1.002),
            "audio_bitrate": "128k",
        }
    return {
        "crop_frac": (0.02, 0.055),
        "scale_jitter": (0.992, 1.008),
        "hue_rad": (-0.25, 0.25),
        "sat_mul": (0.92, 1.12),
        "brightness": (-0.07, 0.07),
        "contrast": (0.94, 1.08),
        "unsharp": ((5, 9), (0.55, 1.25)),
        "noise": (3, 11),
        "crf": (22, 28),
        "preset": ("faster", "fast", "medium"),
        "gop": (48, 240),
        "af_vol": (-2.5, 2.5),
        "atempo": (0.995, 1.005),
        "audio_bitrate": "160k",
    }


def uniqulize_video(input_path: Path, output_path: Path, options: UniqulizeOptions) -> None:
    if VIDEO_DYNAMICS_TEMPORARILY_DISABLED:
        logger.warning(
            "Внимание: уникализация видео с «динамикой» временно недоступна. "
            "Будет выполнена только стерильная пересборка MP4 (очистка метаданных/таймкодов/потоков)."
        )
        strip_all_metadata_file(input_path, output_path)
        return
    ffmpeg = _require_ffmpeg()
    ffprobe = _require_ffprobe()
    rng = _rng_for_bytes(input_path.read_bytes())
    strong = _is_strong(options)
    pv = _video_params(strong)
    logger.info("uniqulize_video: start strong=%s input=%s", strong, input_path.name)

    w, h = _probe_video_size(input_path, ffprobe, ffmpeg)
    if w <= 0 or h <= 0:
        w, h = 1280, 720

    # Геометрический шум по ТЗ: микро-кроп 1–2% + поворот 0.3–0.7°.
    # Кроп делаем первым (центр), чтобы после поворота вероятность чёрных углов была минимальной.
    crop_f = rng.uniform(0.01, 0.02)
    rot_abs = rng.uniform(0.3, 0.7)
    rot_deg = rot_abs if rng.random() < 0.5 else -rot_abs
    rot_rad = rot_deg * (3.141592653589793 / 180.0)
    logger.info("uniqulize_video: geom crop_f=%.4f rotate_deg=%.3f", crop_f, rot_deg)

    cw = max(2, int(w * (1.0 - crop_f)) // 2 * 2)
    ch = max(2, int(h * (1.0 - crop_f)) // 2 * 2)
    cx = (w - cw) // 2
    cy = (h - ch) // 2

    sj = rng.uniform(*pv["scale_jitter"])
    h_rad = rng.uniform(*pv["hue_rad"])
    s_mul = rng.uniform(*pv["sat_mul"])
    # Гистограмма/цвет по ТЗ: динамический сдвиг ±0.02 для brightness/contrast/gamma каждый запуск.
    # brightness у eq — аддитивный (0 = без изменений), contrast/gamma — мультипликативные (1 = без изменений).
    eq_b = rng.uniform(-0.02, 0.02)
    eq_c = 1.0 + rng.uniform(-0.02, 0.02)
    eq_g = 1.0 + rng.uniform(-0.02, 0.02)
    logger.info("uniqulize_video: eq brightness=%.4f contrast=%.4f gamma=%.4f", eq_b, eq_c, eq_g)
    ms_lo, ms_hi = pv["unsharp"][0]
    msize = int(rng.randint(ms_lo, ms_hi))
    if msize % 2 == 0:
        msize += 1
    luma_amt = rng.uniform(*pv["unsharp"][1])
    noise = rng.randint(pv["noise"][0], pv["noise"][1])

    vf_parts = [
        f"crop={cw}:{ch}:{cx}:{cy}",
        f"scale=trunc(iw*{sj}/2)*2:trunc(ih*{sj}/2)*2",
        f"rotate={rot_rad:.12f}:ow=iw:oh=ih:c=black",
        f"hue=h={h_rad}:s={s_mul}",
        f"eq=brightness={eq_b:.6f}:contrast={eq_c:.6f}:gamma={eq_g:.6f}",
        f"unsharp={msize}:{msize}:{luma_amt}:{msize}:{msize}:0.0",
        f"noise=alls={noise}:allf=t",
        "noise=lows=10:flags=p",
    ]
    if rng.random() < (0.18 if strong else 0.08):
        vf_parts.append("hflip")
    base_vf = ",".join(vf_parts)
    n_overlay = rng.randint(1, 4)
    # Дробная частота кадров по ТЗ: сбиваем тайминги (≈ входной FPS * 0.999).
    # Если не смогли определить — берём типовую 30 и тоже сбиваем.
    in_fps = _probe_video_fps(input_path, ffprobe) if ffprobe else 0.0
    if in_fps <= 0:
        in_fps = 30.0
    fps_scale = 0.999 + rng.uniform(-0.0002, 0.0002)
    out_fps = max(8.0, min(120.0, in_fps * fps_scale))
    output_r = f"{out_fps:.3f}"
    fps_f = None
    logger.info("uniqulize_video: fps in=%.3f scale=%.6f out=%s", in_fps, fps_scale, output_r)
    # Невидимый слой шума ~0.005 через overlay (полупрозрачный RGBA поверх основы).
    ovl = (
        f"{base_vf},split=2[main][tmp];"
        f"[tmp]noise=alls={n_overlay}:allf=t,format=rgba,colorchannelmixer=aa=0.005[ovl];"
        f"[main][ovl]overlay=shortest=1:format=yuv420p"
    )
    if fps_f:
        vf = f"{ovl},{fps_f}"
    else:
        vf = ovl

    crf = rng.randint(pv["crf"][0], pv["crf"][1])
    preset = rng.choice(pv["preset"])
    gop = rng.randint(pv["gop"][0], pv["gop"][1])
    x264_refs = rng.choice([2, 3, 3, 4])
    x264_b = rng.choice([2, 3, 3])
    x264_deblock = rng.choice([-1, 0, 0, 1])
    x264p = f"ref={x264_refs}:bframes={x264_b}:deblock={x264_deblock}"

    has_audio = _has_audio_stream(input_path, ffprobe)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    venc = [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        x264p,
        "-bsf:v",
        "h264_metadata=delete_filler=1",
    ]

    def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    af_vol = rng.uniform(*pv["af_vol"])
    # Аудио-энтропия по ТЗ:
    # 1) приводим в 44100,
    # 2) микросдвиг частоты через asetrate (±0.1%),
    # 3) ресемплим обратно в 44100,
    # 4) компенсируем длительность через atempo (обратно фактору), чтобы на слух было максимально незаметно.
    sr_factor = 1.0 + rng.uniform(-0.001, 0.001)
    atempo = 1.0 / sr_factor
    logger.info("uniqulize_video: audio sr_factor=%.6f atempo=%.6f", sr_factor, atempo)
    ch_a = _probe_audio_channels(input_path, ffprobe) if has_audio else 1
    if has_audio:
        if rng.random() < 0.5:
            aud_m = f"aecho=0.8:0.9:{rng.randint(10, 28)}:{rng.uniform(0.15, 0.35):.3f}"
        else:
            if ch_a >= 2:
                aud_m = "pan=stereo|c0=-c0|c1=-c1"
            else:
                aud_m = f"aecho=0.8:0.88:{rng.randint(12, 26)}:{rng.uniform(0.16, 0.32):.3f}"
        af = (
            "aresample=44100,"
            f"asetrate=44100*{sr_factor:.6f},"
            "aresample=44100,"
            f"{aud_m},"
            f"volume={af_vol:.4f}dB,"
            f"atempo={atempo:.6f}"
        )
    else:
        af = ""

    out_r_args: list[str] = ["-r", str(output_r)] if output_r else []

    cmd_base = [
        ffmpeg,
        "-bitexact",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+bitexact",
        "-probesize",
        "50M",
        "-analyzeduration",
        "100M",
        "-i",
        str(input_path),
        "-avoid_negative_ts",
        "make_zero",
    ] + _ffmpeg_sterile_mp4_metadata_args() + [
        "-dn",
        "-sn",
        "-vf",
        vf,
    ]

    if has_audio:
        cmd = cmd_base + ["-af", af] + venc + [
            "-c:a",
            "aac",
            "-b:a",
            pv["audio_bitrate"],
            "-ar",
            "44100",
        ] + out_r_args + [str(output_path)]
    else:
        cmd = cmd_base + ["-an"] + venc + out_r_args + [str(output_path)]

    proc = _run(cmd)
    if proc.returncode != 0 and has_audio:
        cmd2 = cmd_base + ["-an"] + venc + out_r_args + ["-movflags", "+faststart", str(output_path)]
        proc2 = _run(cmd2)
        if proc2.returncode != 0:
            err = (proc2.stderr or proc.stderr or "")[-2500:]
            raise RuntimeError(f"ffmpeg ошибка:\n{err}")
        return
    if proc.returncode != 0:
        err = (proc.stderr or "")[-2500:]
        raise RuntimeError(f"ffmpeg ошибка (код {proc.returncode}):\n{err}")
    logger.info("uniqulize_video: done output=%s", output_path.name)


def _probe_video_size(input_path: Path, ffprobe: str | None, ffmpeg: str) -> tuple[int, int]:
    if ffprobe:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(input_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            m = re.match(r"^(\d+)x(\d+)", r.stdout.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
    try:
        r2 = subprocess.run(
            [ffmpeg, "-i", str(input_path), "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        if getattr(e, "winerror", None) == 4551:
            raise RuntimeError(
                f"Windows заблокировал запуск ffmpeg: {ffmpeg}. "
                "Установи доверенный ffmpeg (winget/choco/официальная сборка) и убедись, что он в PATH, "
                "либо укажи полный путь через UNIQUILIZER_FFMPEG/FFMPEG_PATH. "
                "Если у тебя лежит старый ffmpeg в data/ffmpeg — удали его."
            ) from e
        raise
    err = r2.stderr or ""
    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _uniq_creo_filename(out_suffix: str) -> str:
    """Имя файла крео: onyx-fb-<уникальный id>.jpg / .mp4"""
    uid = secrets.token_hex(8)
    return f"onyx-fb-{uid}{out_suffix.lower()}"


def uniqulize_file(input_path: Path, options: UniqulizeOptions) -> Path:
    """Пишет результат рядом с входом (в той же рабочей папке); временный `_*` удаляется после копии."""
    suffix = input_path.suffix.lower()
    work = input_path.parent
    if suffix in IMAGE_INPUT_SUFFIXES:
        out = work / "_uniqulizer_out.jpg"
        uniqulize_image(input_path, out, options)
    elif suffix in VIDEO_INPUT_SUFFIXES:
        out = work / "_uniqulizer_out.mp4"
        uniqulize_video(input_path, out, options)
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    final = work / _uniq_creo_filename(out.suffix)
    final.write_bytes(out.read_bytes())
    try:
        out.unlink(missing_ok=True)
    except OSError:
        pass
    return final


def process_file_stateless(input_path: Path, *, mode: str = "uniqulize", options: UniqulizeOptions | None = None, deep: DeepAugmentOptions | None = None) -> tuple[bytes, str]:
    """Stateless обработка: вход -> временная директория -> bytes результата -> удаление файлов.

    Возвращает (output_bytes, suggested_filename).
    """
    suffix = input_path.suffix.lower()
    data = input_path.read_bytes()
    return process_bytes_stateless(data, suffix=suffix, mode=mode, options=options, deep=deep)


def process_bytes_stateless(
    data: bytes,
    *,
    suffix: str,
    mode: str = "uniqulize",
    options: UniqulizeOptions | None = None,
    deep: DeepAugmentOptions | None = None,
) -> tuple[bytes, str]:
    """Stateless обработка bytes.

    - Ничего не сохраняет «на постоянку» (кроме ffmpeg-бинарников, если включена автозагрузка).
    - Все промежуточные/финальные файлы лежат в temp и удаляются до возврата.
    """
    s = (suffix or "").lower().strip()
    if not s.startswith("."):
        s = "." + s if s else ".bin"

    if mode not in ("uniqulize", "deep"):
        raise ValueError("mode должен быть 'uniqulize' или 'deep'")

    # Защита: в stateless-режиме по умолчанию не добавляем никакие синтетические EXIF.
    if options is None:
        options = UniqulizeOptions()
    if deep is None:
        deep = DeepAugmentOptions()

    safe_options = options
    if getattr(safe_options, "synthetic_exif", True):
        safe_options = UniqulizeOptions(
            preset=safe_options.preset,
            preserve_visual=safe_options.preserve_visual,
            synthetic_exif=False,
        )

    td = tempfile.mkdtemp(prefix="uniqulizer_")
    tdir = Path(td)
    in_path = tdir / ("input" + s)
    out_path: Path | None = None
    try:
        in_path.write_bytes(data)

        if s in IMAGE_INPUT_SUFFIXES:
            out_suffix = ".jpg"
        elif s in VIDEO_INPUT_SUFFIXES:
            out_suffix = ".mp4"
        else:
            raise ValueError(f"Неподдерживаемый формат: {s}")

        out_path = tdir / ("output" + out_suffix)

        if mode == "uniqulize":
            if s in IMAGE_INPUT_SUFFIXES:
                uniqulize_image(in_path, out_path, safe_options)
            else:
                uniqulize_video(in_path, out_path, safe_options)
        else:
            deep_augment_file(in_path, out_path, deep)

        out_bytes = out_path.read_bytes()
        name = _uniq_creo_filename(out_suffix)
        return out_bytes, name
    finally:
        for p in (out_path, in_path):
            if p is None:
                continue
            try:
                if p.exists():
                    os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tdir)
        except OSError:
            try:
                shutil.rmtree(tdir, ignore_errors=True)
            except Exception:
                pass
