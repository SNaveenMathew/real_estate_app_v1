"""Upload handling shared by every multipart endpoint.

The service binds ``0.0.0.0`` (``config.settings.app_host``), so a multipart body is
untrusted input. Three rules are enforced here instead of in each endpoint:

1. **A client filename is a name, never a path.** ``sanitize_filename`` keeps only the
   last path segment, reduces it to a conservative character set and never returns an
   empty, dot-only or Windows-reserved name - so ``"../../secrets.csv"`` becomes
   ``secrets.csv`` and the result is always safe to join onto a directory.
2. **Nothing is written past a hard byte cap**, and a rejected upload leaves no
   partial file behind (``save_stream``).
3. **An image's type comes from its bytes.** ``sniff_image_extension`` decodes the
   file with Pillow and derives the extension from the detected format, so a script
   named ``photo.png`` is rejected instead of stored and served back.

Callers still decide which extensions they accept; this module only guarantees the
name is safe and the bytes are what they claim to be.

Blocking by design: these are plain file operations. An ``async def`` endpoint must
call them through ``asyncio.to_thread`` (or be a plain ``def`` handler, which FastAPI
already runs in its worker pool) rather than on the event loop.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException

CHUNK_SIZE = 1024 * 1024
MAX_FILENAME_STEM = 100

# Windows treats these as device names even with an extension ("CON.csv" fails to open).
_RESERVED_STEMS = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

# Pillow format -> extension written to disk. The extension is derived from the decoded
# bytes, so it can never disagree with the content.
IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}


def sanitize_filename(raw: str | None, *, fallback: str = "upload") -> str:
    """A single safe basename derived from a client-supplied filename.

    ``"..\\..\\etc\\passwd"`` -> ``"passwd"``; ``".csv"`` -> ``"csv"`` (no extension, so
    the caller's own allow-list rejects it); ``"CON.csv"`` -> ``"_CON.csv"``.
    """
    name = unicodedata.normalize("NFKC", str(raw or "")).replace("\\", "/")
    name = name.rsplit("/", 1)[-1]                                   # basename only: no traversal, no drive letters
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")       # no control chars, quotes, or shell metacharacters
    stem, dot, suffix = name.rpartition(".")
    if not dot or not stem:                                          # "csv", ".csv", "..." have no usable stem
        stem, suffix = name, ""
    stem = stem.strip(" .")[:MAX_FILENAME_STEM] or fallback
    if stem.upper() in _RESERVED_STEMS:
        stem = f"_{stem}"
    return f"{stem}.{suffix}" if suffix else stem


def save_stream(src: BinaryIO, dest: Path, *, max_bytes: int) -> int:
    """Copy ``src`` to ``dest`` in chunks, refusing more than ``max_bytes``.

    Returns the number of bytes written. On overflow (or an empty upload) the partial
    file is removed and ``HTTPException`` is raised, so a rejected request never leaves
    a truncated file behind. Blocking; see the module docstring.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(dest, "wb") as fh:
            while chunk := src.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"The file is larger than {max_bytes // (1024 * 1024)} MB.",
                    )
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    return size


def sniff_image_extension(path: Path) -> str:
    """The extension implied by the image bytes, or 415 if this is not an image we store.

    Format detection and decoding are Pillow's job: ``verify()`` walks the whole file,
    so a well-formed header on a truncated or crafted payload still fails here.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            img.verify()
    except Exception:
        raise HTTPException(
            status_code=415,
            detail="That file is not a readable image (JPEG, PNG, WebP or GIF).",
        )
    ext = IMAGE_FORMATS.get(fmt)
    if ext is None:
        raise HTTPException(
            status_code=415,
            detail=f"'{fmt.lower() or 'that'}' images are not supported. Use JPEG, PNG, WebP or GIF.",
        )
    return ext
