"""
Region compositing for the iterative per-person edit loop.

After each edit pass we paste back ONLY the edited person's bbox region over the
working image, so untouched people stay pixel-exact and cumulative drift can't
build up across many passes (spec §3, §7). A few-pixel feather on the mask edge
avoids visible seams.

Pure PIL/numpy — no GPU, no ComfyUI. The worker base image already has both.
"""

import io
import base64
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def load_image(path_or_pil):
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    return Image.open(path_or_pil).convert("RGB")


def scale_to_megapixels(img: Image.Image, megapixels: float = 1.0) -> Image.Image:
    """Match the editor graph's ImageScaleToTotalPixels (node 93) so the planner
    sees the same pixel space the editor renders in, keeping bboxes aligned."""
    w, h = img.size
    target = megapixels * 1_000_000
    if w * h <= 0:
        return img
    scale = (target / float(w * h)) ** 0.5
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)


def b64_to_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clamp_bbox(bbox, size):
    """Clamp [x, y, w, h] to image bounds; return integer (x0, y0, x1, y1)."""
    w_img, h_img = size
    x, y, w, h = bbox
    x0 = max(0, min(int(round(x)), w_img - 1))
    y0 = max(0, min(int(round(y)), h_img - 1))
    x1 = max(x0 + 1, min(int(round(x + w)), w_img))
    y1 = max(y0 + 1, min(int(round(y + h)), h_img))
    return x0, y0, x1, y1


def feathered_mask(size, bbox, feather: int = 6) -> np.ndarray:
    """0..1 float HxW mask: 1 inside the bbox, fading to 0 over `feather` px."""
    w_img, h_img = size
    x0, y0, x1, y1 = _clamp_bbox(bbox, size)
    m = Image.new("L", (w_img, h_img), 0)
    m.paste(255, (x0, y0, x1, y1))
    if feather and feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return (np.asarray(m).astype(np.float32) / 255.0)


def composite(base: Image.Image, edited: Image.Image, bbox, feather: int = 6) -> Image.Image:
    """composite = base*(1-mask) + edited*mask, with a feathered bbox mask.

    `edited` is resized to base's size first — the editor rescales to ~1MP and may
    return slightly different dims, so we align defensively before blending.
    """
    base = base.convert("RGB")
    edited = edited.convert("RGB")
    if edited.size != base.size:
        edited = edited.resize(base.size, Image.LANCZOS)
    m = feathered_mask(base.size, bbox, feather)[..., None]  # HxWx1
    b = np.asarray(base).astype(np.float32)
    e = np.asarray(edited).astype(np.float32)
    out = b * (1.0 - m) + e * m
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")
