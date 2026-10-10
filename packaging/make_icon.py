"""Writes packaging/foresight.ico: a near-black rounded square with the recording dot."""
import struct
import zlib
from pathlib import Path

import numpy as np


def render(size: int) -> np.ndarray:
    ss = 4  # supersample for anti-aliasing
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    r = 0.22
    dx = np.maximum(np.abs(x - 0.5) - (0.5 - r), 0)
    dy = np.maximum(np.abs(y - 0.5) - (0.5 - r), 0)
    square = (dx ** 2 + dy ** 2) <= r ** 2
    dot = (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.2 ** 2
    img = np.zeros((n, n, 4))
    img[square] = (0x14, 0x14, 0x14, 255)
    img[dot] = (0xD7, 0x38, 0x1B, 255)
    img = img.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    # un-premultiply colour where alpha is partial
    a = img[..., 3:4]
    img[..., :3] = np.where(a > 0, img[..., :3] * 255 / np.maximum(a, 1e-6), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


sizes = [16, 24, 32, 48, 64, 128, 256]
images = [png(render(s)) for s in sizes]
header = struct.pack("<HHH", 0, 1, len(sizes))
offset = 6 + 16 * len(sizes)
entries = b""
for s, data in zip(sizes, images):
    entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
out = Path(__file__).with_name("foresight.ico")
out.write_bytes(header + entries + b"".join(images))
Path(__file__).with_name("foresight-256.png").write_bytes(images[-1])
print(out, out.stat().st_size, "bytes")
