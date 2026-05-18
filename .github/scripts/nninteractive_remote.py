"""
Remote nnInteractive backend over HTTP (SlicerNNInteractive server).

This module implements ``RemoteSegmenter``, a drop-in replacement for the
local :class:`nninteractive_segment.Segmenter`. It speaks the public HTTP
API of the upstream SlicerNNInteractive FastAPI server
(https://github.com/coendevente/SlicerNNInteractive), so we don't ship
our own server code at all.

Why
---
Heavy nnU-Net 3D inference doesn't fit in a 16 GB Mac mini's RAM
(empirically peaks at ~10.5 GB on a 119^3 volume at 1 thread; >16 GB at
4 threads -> SIGKILL). The SlicerNNInteractive server runs the model on
a beefier box (typically a Jetstream2 / MorphoCloud GPU instance) and
exposes a tiny FastAPI surface over HTTP. The Exosphere any-port
HTTPS reverse proxy in front of MorphoCloud means we don't need SSH or
a tunnel — we just point requests at
``https://http-<ip-with-dashes>-<port>.proxy-js2-iu.exosphere.app/``.

Wire protocol (the upstream server's, verbatim)
-----------------------------------------------
* ``POST /upload_image``
    multipart ``file=*.npy`` (uncompressed)
    body: ``np.save(buf, arr)`` where ``arr`` has shape (z, y, x).
    reply: ``{"status": "ok"}``

* ``POST /upload_segment``
    multipart ``file=*.npy.gz`` (gzipped) — used here to reset the
    segmentation by uploading an all-zeros mask.

* ``POST /add_point_interaction``
    JSON ``{"voxel_coord": [x, y, z], "positive_click": bool}``
    reply: gzipped 1-bit-per-voxel packed mask (Content-Encoding: gzip)

* ``POST /add_bbox_interaction``
    JSON ``{"outer_point_one": [x0, y0, z0],
            "outer_point_two": [x1, y1, z1],
            "positive_click": bool}``
    reply: gzipped 1-bit-per-voxel packed mask

* ``POST /add_lasso_interaction`` / ``/add_scribble_interaction``
    multipart ``file=mask.npy.gz`` + form ``positive_click=str``
    (not used by the current paint loop)

The mask is unpacked locally with ``np.unpackbits`` and reshaped to the
volume's (z, y, x) shape.

Notes
-----
* Rendering of orthogonal previews and saving of the final NIfTI labelmap
  happens locally, using the original volume's SimpleITK geometry. The
  remote box never needs SimpleITK or matplotlib.
* The upstream server has no built-in auth; the Exosphere proxy URL
  itself is the secret. Set ``NNI_REMOTE_URL`` in your local ``.env`` and
  treat the URL like a token. (For old envs that still have
  ``NNI_REMOTE_WS=wss://…``, the URL is auto-translated to https://.)
* Uses the local ``requests`` library; falls back to a small
  ``urllib``-based pure-stdlib client if requests isn't available, so
  that we don't bring an extra dependency into the parent venv.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk

log = logging.getLogger("nninteractive_remote_http")


# ---------------------------------------------------------------------------
# Tiny HTTP shim: prefer requests, fall back to urllib so we stay
# dependency-light when running from the parent venv.
# ---------------------------------------------------------------------------

try:
    import requests as _requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    _HAS_REQUESTS = False
    _requests = None


@dataclass
class _Resp:
    status: int
    headers: dict
    content: bytes

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self):
        if not (200 <= self.status < 300):
            preview = self.content[:300].decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {self.status}: {preview}")


def _post_json(url: str, payload: dict, timeout: float) -> _Resp:
    if _HAS_REQUESTS:
        r = _requests.post(url, json=payload, timeout=timeout)
        return _Resp(r.status_code, dict(r.headers), r.content)
    import urllib.request
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _Resp(resp.status, dict(resp.headers), resp.read())


def _post_multipart(url: str, file_field: str, file_bytes: bytes,
                    file_name: str, timeout: float,
                    extra_form: Optional[dict] = None) -> _Resp:
    if _HAS_REQUESTS:
        files = {file_field: (file_name, file_bytes, "application/octet-stream")}
        r = _requests.post(
            url, files=files, data=extra_form or {}, timeout=timeout,
        )
        return _Resp(r.status_code, dict(r.headers), r.content)

    # Hand-rolled multipart for the urllib fallback. Boundary is fixed
    # per-call but unique enough; keep it simple.
    import urllib.request
    boundary = f"----nni-{int(time.time()*1e6):x}"
    parts: list[bytes] = []
    if extra_form:
        for k, v in extra_form.items():
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            )
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _Resp(resp.status, dict(resp.headers), resp.read())


def _get(url: str, timeout: float) -> _Resp:
    if _HAS_REQUESTS:
        r = _requests.get(url, timeout=timeout)
        return _Resp(r.status_code, dict(r.headers), r.content)
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return _Resp(resp.status, dict(resp.headers), resp.read())


# ---------------------------------------------------------------------------
# Prompt log entry — mirrors local Segmenter
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    kind: str
    coords: object = None
    include: bool = True
    label: str = ""


# ---------------------------------------------------------------------------
# RemoteSegmenter — drop-in replacement for nninteractive_segment.Segmenter
# ---------------------------------------------------------------------------

class RemoteSegmenter:
    """HTTP-based RemoteSegmenter that proxies to a SlicerNNInteractive server."""

    def __init__(self, config):
        self.config = config
        self.input_path = Path(config.input_path)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = self._read_base_url()
        self.timeout = float(os.environ.get("NNI_REMOTE_TIMEOUT", "300"))
        log.info("RemoteSegmenter -> %s (timeout=%.0fs, requests=%s)",
                 self.base_url, self.timeout, _HAS_REQUESTS)

        # ---- 1. load the volume locally so previews/labelmap keep the
        #         original SimpleITK geometry (origin, spacing, direction).
        log.info("Loading volume: %s", self.input_path)
        self.sitk_image = sitk.ReadImage(str(self.input_path))
        arr = sitk.GetArrayFromImage(self.sitk_image)  # (z, y, x)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume; got shape {arr.shape} from {self.input_path}"
            )
        self.image_shape_zyx = tuple(arr.shape)

        # ---- 2. sanity-check the server is reachable. We probe /docs,
        #         which FastAPI serves as 200 (Swagger UI) once the
        #         uvicorn worker is up.
        try:
            docs = _get(self.base_url + "/docs", timeout=10)
            log.info("Server probe: GET /docs -> HTTP %s", docs.status)
            if docs.status >= 500:
                raise RuntimeError(
                    f"Server probe returned HTTP {docs.status}; "
                    "is nninteractive-slicer-server bound to 0.0.0.0?"
                )
        except Exception as e:
            raise RuntimeError(
                f"Could not reach SlicerNNInteractive server at {self.base_url}: {e}\n"
                f"  - Verify the server is running on the box (port matches the URL).\n"
                f"  - Verify it is bound to 0.0.0.0, not 127.0.0.1, so the\n"
                f"    Exosphere proxy can reach it."
            ) from e

        # ---- 3. upload the volume once. Server resets interactions on upload.
        self._upload_image(arr)

        # ---- 4. local mirror of the segmentation mask (server returns
        #         the full mask after every interaction).
        self._mask = np.zeros(self.image_shape_zyx, dtype=np.uint8)
        self._last_step_seconds: Optional[float] = None
        self._last_mask_bytes: Optional[int] = None

        self.prompts_log: list[Prompt] = []

    # ------------------------------------------------------------------
    # Misc properties to look like a local Segmenter
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return f"remote ({self.base_url})"

    @property
    def model_path(self) -> str:
        return f"remote: nnInteractive (SlicerNNInteractive @ {self.base_url})"

    @staticmethod
    def _read_base_url() -> str:
        url = (
            os.environ.get("NNI_REMOTE_URL", "").strip()
            or os.environ.get("NNI_REMOTE_WS", "").strip()
        )
        if not url:
            raise RuntimeError(
                "NNI_REMOTE_URL is not set. Add it to your .env, e.g.\n"
                "  NNI_REMOTE_URL=https://http-149-165-172-55-1527.proxy-js2-iu.exosphere.app/"
            )
        # Translate ws[s]://host -> http[s]://host so envs that still
        # carry the old NNI_REMOTE_WS=wss://… keep working.
        if url.startswith("ws://"):
            url = "http://" + url[len("ws://"):]
        elif url.startswith("wss://"):
            url = "https://" + url[len("wss://"):]
        return url.rstrip("/")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _upload_image(self, arr: np.ndarray) -> None:
        """POST /upload_image with a npy-encoded volume (server expects (z,y,x))."""
        log.info("Uploading volume to remote (shape=%s, dtype=%s)…",
                 arr.shape, arr.dtype)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        payload = buf.getvalue()
        t0 = time.time()
        r = _post_multipart(
            self.base_url + "/upload_image",
            file_field="file", file_bytes=payload,
            file_name="volume.npy", timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "ok":
            raise RuntimeError(f"upload_image failed: {body}")
        log.info("Volume uploaded in %.1f s (%.2f MB)",
                 time.time() - t0, len(payload) / 1e6)

    def _decode_mask(self, content: bytes) -> np.ndarray:
        """Decode the gzipped 1-bit-per-voxel mask returned by the server."""
        try:
            decompressed = gzip.decompress(content)
        except OSError:
            # Server may already have applied Content-Encoding: gzip and
            # the HTTP layer auto-decoded it. Treat content as raw bytes.
            decompressed = content
        bits = np.unpackbits(np.frombuffer(decompressed, dtype=np.uint8))
        n = int(np.prod(self.image_shape_zyx))
        if bits.size < n:
            raise RuntimeError(
                f"server returned only {bits.size} bits, expected at least {n}"
            )
        mask = bits[:n].reshape(self.image_shape_zyx).astype(np.uint8)
        self._last_mask_bytes = len(content)
        return mask

    @staticmethod
    def _is_json_error(headers: dict, content: bytes) -> Optional[dict]:
        ctype = (headers.get("Content-Type") or headers.get("content-type") or "")
        if "application/json" not in ctype.lower():
            return None
        try:
            body = json.loads(content.decode("utf-8"))
        except Exception:
            return None
        if isinstance(body, dict) and body.get("status") == "error":
            return body
        return None

    # ------------------------------------------------------------------
    # Prompt API (mirrors local Segmenter)
    # ------------------------------------------------------------------

    def add_point(self, x: int, y: int, z: int, *, positive: bool = True,
                  label: str = "") -> None:
        log.info("[remote] add_point(x=%d, y=%d, z=%d, positive=%s)",
                 x, y, z, positive)
        t0 = time.time()
        r = _post_json(
            self.base_url + "/add_point_interaction",
            {"voxel_coord": [int(x), int(y), int(z)],
             "positive_click": bool(positive)},
            timeout=self.timeout,
        )
        r.raise_for_status()
        err = self._is_json_error(r.headers, r.content)
        if err:
            raise RuntimeError(f"server error on add_point: {err.get('message')}")
        self._mask = self._decode_mask(r.content)
        self._last_step_seconds = time.time() - t0
        self.prompts_log.append(
            Prompt(kind="point", coords=(int(x), int(y), int(z)),
                   include=bool(positive), label=label)
        )
        log.info("[remote] mask voxel_count=%d  (step %.1fs, payload %.1f kB)",
                 int((self._mask > 0).sum()),
                 self._last_step_seconds,
                 (self._last_mask_bytes or 0) / 1024)

    def add_bbox(self, x_range: Iterable[int], y_range: Iterable[int],
                 z_range: Iterable[int], *, positive: bool = True,
                 label: str = "") -> None:
        x0, x1 = int(x_range[0]), int(x_range[1])
        y0, y1 = int(y_range[0]), int(y_range[1])
        z0, z1 = int(z_range[0]), int(z_range[1])
        log.info("[remote] add_bbox(x=[%d,%d] y=[%d,%d] z=[%d,%d], positive=%s)",
                 x0, x1, y0, y1, z0, z1, positive)
        t0 = time.time()
        r = _post_json(
            self.base_url + "/add_bbox_interaction",
            {"outer_point_one": [x0, y0, z0],
             "outer_point_two": [x1, y1, z1],
             "positive_click": bool(positive)},
            timeout=self.timeout,
        )
        r.raise_for_status()
        err = self._is_json_error(r.headers, r.content)
        if err:
            raise RuntimeError(f"server error on add_bbox: {err.get('message')}")
        self._mask = self._decode_mask(r.content)
        self._last_step_seconds = time.time() - t0
        self.prompts_log.append(
            Prompt(kind="bbox",
                   coords=[[x0, x1], [y0, y1], [z0, z1]],
                   include=bool(positive), label=label)
        )
        log.info("[remote] mask voxel_count=%d  (step %.1fs, payload %.1f kB)",
                 int((self._mask > 0).sum()),
                 self._last_step_seconds,
                 (self._last_mask_bytes or 0) / 1024)

    def reset_segment(self) -> None:
        """Reset by uploading an all-zeros mask via /upload_segment.

        SlicerNNInteractive's PromptManager.set_segment() resets
        interactions when given an all-zero mask, which is the documented
        way to start over without re-uploading the volume.
        """
        log.info("[remote] reset_segment")
        empty = np.zeros(self.image_shape_zyx, dtype=np.uint8)
        buf = io.BytesIO()
        np.save(buf, empty, allow_pickle=False)
        compressed = gzip.compress(buf.getvalue())
        r = _post_multipart(
            self.base_url + "/upload_segment",
            file_field="file", file_bytes=compressed,
            file_name="seg.npy.gz", timeout=self.timeout,
        )
        r.raise_for_status()
        self._mask[:] = 0
        self.prompts_log.append(Prompt(kind="reset"))

    # ------------------------------------------------------------------
    # Result accessors (mirrors local Segmenter)
    # ------------------------------------------------------------------

    @property
    def mask_array(self) -> np.ndarray:
        return self._mask

    def voxel_count(self) -> int:
        return int((self._mask > 0).sum())

    def volume_mm3(self) -> float:
        spacing = self.sitk_image.GetSpacing()  # (sx, sy, sz)
        voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
        return self.voxel_count() * voxel_vol

    def bounding_box(self) -> Optional[dict]:
        if self._mask.sum() == 0:
            return None
        zs, ys, xs = np.where(self._mask > 0)
        return {
            "x": [int(xs.min()), int(xs.max())],
            "y": [int(ys.min()), int(ys.max())],
            "z": [int(zs.min()), int(zs.max())],
        }

    # ------------------------------------------------------------------
    # Persistence — performed locally so we keep the original geometry
    # ------------------------------------------------------------------

    def save_labelmap(self, name: str = "") -> str:
        out_name = name or f"{self.config.media_id}_nni_labelmap.nii.gz"
        out_path = self.output_dir / out_name
        seg_image = sitk.GetImageFromArray(self._mask.astype(np.uint8))
        seg_image.CopyInformation(self.sitk_image)
        sitk.WriteImage(seg_image, str(out_path), useCompression=True)
        log.info("Wrote labelmap: %s (%d voxels, %.2f mm^3)",
                 out_path, self.voxel_count(), self.volume_mm3())
        return str(out_path)

    def save_orthogonal_previews(self, name_prefix: str = "") -> list[str]:
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed; skipping preview screenshots")
            return []

        arr = sitk.GetArrayFromImage(self.sitk_image)  # (z, y, x)
        mask = self._mask
        prefix = name_prefix or f"{self.config.media_id}_nni"

        def _best_slice(axis: int) -> int:
            if mask.sum() == 0:
                return arr.shape[axis] // 2
            sums = mask.sum(axis=tuple(i for i in range(3) if i != axis))
            return int(np.argmax(sums))

        previews: list[str] = []
        views = [
            ("axial",    0, lambda v, s: (v[s, :, :], mask[s, :, :])),
            ("coronal",  1, lambda v, s: (v[:, s, :], mask[:, s, :])),
            ("sagittal", 2, lambda v, s: (v[:, :, s], mask[:, :, s])),
        ]
        for view_name, axis, slicer in views:
            s = _best_slice(axis)
            img_slice, mask_slice = slicer(arr, s)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_slice, cmap="gray", origin="lower")
            if mask_slice.sum() > 0:
                overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(overlay, cmap="autumn", origin="lower",
                          alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"{prefix} {view_name} idx={s}")
            ax.axis("off")
            out_path = self.output_dir / f"{prefix}_{view_name}.png"
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            plt.close(fig)
            previews.append(str(out_path))
        return previews

    def export_summary(self, extra: Optional[dict] = None) -> str:
        out_path = self.output_dir / f"{self.config.media_id}_nni_summary.json"
        summary = {
            "media_id": self.config.media_id,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "remote_url": self.base_url,
            "image_shape_zyx": list(self.image_shape_zyx),
            "voxel_count": self.voxel_count(),
            "volume_mm3": round(self.volume_mm3(), 3),
            "bounding_box_voxels": self.bounding_box(),
            "prompts": [
                {"kind": p.kind, "coords": p.coords,
                 "include": p.include, "label": p.label}
                for p in self.prompts_log
            ],
            "last_step_seconds": self._last_step_seconds,
        }
        if extra:
            summary.update(extra)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        log.info("Wrote summary: %s", out_path)
        return str(out_path)

    def close(self) -> None:
        # No persistent connection; nothing to clean up.
        pass


# Backwards-compat aliases for callers that imported the old WS-based
# RemoteSegmenter or used the make_remote_segmenter() factory name.
def make_remote_segmenter(config) -> RemoteSegmenter:  # pragma: no cover
    return RemoteSegmenter(config)
