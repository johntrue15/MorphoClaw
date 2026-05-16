"""
Remote nnInteractive backend over WebSocket.

This module implements ``RemoteSegmenter``, a drop-in replacement for the
local :class:`nninteractive_segment.Segmenter`. It speaks to an
``nni_ws_server.py`` instance running on a beefier machine (e.g. a
MorphoCloud / Jetstream2 VM) and proxies every prompt over a single
persistent WebSocket connection.

Why
---
Heavy nnU-Net 3D inference doesn't fit in a 16 GB Mac mini's RAM
(empirically peaks at ~10.5 GB on a 119^3 volume at 1 thread; >16 GB at
4 threads -> SIGKILL). Running the model on a 30+ GB cloud box, with the
orchestration loop and OPENAI_API_KEY still on the local machine, is the
clean fix.

Wire protocol (v1)
------------------
The connection is single-session: one client manages one volume at a
time. Frames are either JSON text or raw binary. Every server reply that
carries bulk data (a mask, a screenshot, a labelmap) is a JSON envelope
followed by exactly one binary frame.

Client -> Server (JSON):
    {"op": "hello", "version": 1}
    {"op": "load_image",
        "shape_zyx": [Z, Y, X],
        "spacing_xyz": [sx, sy, sz],
        "origin_xyz": [ox, oy, oz],
        "direction": [d0..d8],
        "dtype": "float32" | "int16" | "uint8",
        "media_id": "...",
        "size_bytes": N,
        "compression": "none" | "gzip"}
    <binary frame: raw volume bytes (zyx order), length size_bytes>

    {"op": "add_point", "x": int, "y": int, "z": int,
                        "positive": bool, "label": "..."}
    {"op": "add_bbox",  "x": [x0,x1], "y": [y0,y1], "z": [z0,z1],
                        "positive": bool, "label": "..."}
    {"op": "reset"}
    {"op": "get_mask", "compression": "gzip"}
    {"op": "info"}
    {"op": "ping"}
    {"op": "close"}

Server -> Client (JSON, optionally followed by one binary frame):
    {"op": "hello_ack", "version": 1, "device": "cuda:0",
        "model": "nnInteractive_v1.0", "model_loaded": true}
    {"op": "loaded", "shape_zyx": [...], "spacing_xyz": [...],
        "duration_s": 4.2}
    {"op": "step_done", "voxel_count": int, "duration_s": float,
        "mask_size_bytes": M, "compression": "gzip"}
    <binary frame: gzipped uint8 mask in zyx order, length mask_size_bytes>
    {"op": "info_reply", "memory_gb": float, ...}
    {"op": "pong"}
    {"op": "error", "message": "...", "context": "..."}

Notes
-----
- The server compresses masks (uint8, mostly zeros) with gzip by default;
  for a 119^3 volume with a few thousand foreground voxels this brings
  the per-step transfer from ~1.7 MB to a few KB.
- The volume is uploaded once at session start; subsequent prompts only
  exchange small JSON + small mask bytes.
- Rendering of orthogonal previews and saving of the final NIfTI labelmap
  happens locally, using the original volume's SimpleITK geometry. The
  remote box never needs SimpleITK or matplotlib.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Defer heavy imports (websocket-client, SimpleITK, numpy, matplotlib) to
# the methods that actually need them so importing this module is cheap
# and so that very small CLIs (e.g. ping the server) don't pay the cost.

log = logging.getLogger("nninteractive.remote")


PROTOCOL_VERSION = 1
DEFAULT_WS_URL = os.environ.get("NNI_REMOTE_WS", "ws://localhost:8765")
DEFAULT_RECV_TIMEOUT_S = float(
    os.environ.get("NNI_REMOTE_RECV_TIMEOUT_S", "600")
)


class RemoteSegmenterError(RuntimeError):
    """Raised on protocol/transport errors against the WS server."""


# Re-export the same Prompt dataclass so RemoteSegmenter looks identical
# to the local Segmenter from the loop's point of view.
from nninteractive_segment import Prompt, SegmenterConfig  # noqa: E402


class RemoteSegmenter:
    """Proxy for the local :class:`Segmenter` API over WebSocket.

    Public surface intentionally mirrors :class:`nninteractive_segment.Segmenter`
    so :func:`nninteractive_loop.run_loop` can be used unchanged.
    """

    def __init__(self, config: SegmenterConfig, ws_url: str = ""):
        try:
            from websocket import create_connection  # type: ignore
        except ImportError as exc:
            raise RemoteSegmenterError(
                "RemoteSegmenter requires the 'websocket-client' package. "
                "Install it with: pip install websocket-client"
            ) from exc

        import numpy as np
        import SimpleITK as sitk

        self._np = np
        self._sitk = sitk

        self.config = config
        self.input_path = Path(config.input_path)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ws_url = ws_url or DEFAULT_WS_URL
        if not self.ws_url:
            raise RemoteSegmenterError(
                "No WebSocket URL configured (set NNI_REMOTE_WS)."
            )

        log.info("Connecting to nnInteractive WS server: %s", self.ws_url)
        self._ws = create_connection(
            self.ws_url, timeout=DEFAULT_RECV_TIMEOUT_S
        )

        # ---- handshake (with optional shared-secret auth)
        token = os.environ.get("NNI_WS_TOKEN", "").strip()
        hello = {"op": "hello", "version": PROTOCOL_VERSION}
        if token:
            hello["token"] = token
        self._send_json(hello)
        ack = self._recv_json()
        if ack.get("op") != "hello_ack":
            raise RemoteSegmenterError(
                f"Unexpected handshake reply: {ack!r}"
            )
        if ack.get("version") != PROTOCOL_VERSION:
            raise RemoteSegmenterError(
                f"Server protocol v{ack.get('version')} != "
                f"client v{PROTOCOL_VERSION}"
            )
        self.remote_device = ack.get("device", "?")
        self.remote_model = ack.get("model", "?")
        log.info("Server ready: device=%s model=%s",
                 self.remote_device, self.remote_model)

        # ---- load the volume LOCALLY (we need its sitk geometry for
        # rendering / saving) and ship the bytes to the server
        log.info("Loading local volume: %s", self.input_path)
        self.sitk_image = sitk.ReadImage(str(self.input_path))
        arr = sitk.GetArrayFromImage(self.sitk_image)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume; got shape {arr.shape} "
                f"from {self.input_path}"
            )
        self.image_shape_zyx = tuple(arr.shape)

        # The mask is what we update step-by-step; start at zero. We keep
        # a local copy so we can render previews without round-tripping.
        self.mask = np.zeros(self.image_shape_zyx, dtype=np.uint8)

        # Upload volume to server (gzip helps for sparse CT, hurts for
        # dense volumes; default off to keep things fast).
        compress_upload = (
            os.environ.get("NNI_REMOTE_COMPRESS_UPLOAD", "0") == "1"
        )
        # We send the volume in its native dtype to avoid surprises.
        upload_arr = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
        upload_bytes = upload_arr.tobytes()
        if compress_upload:
            upload_bytes = gzip.compress(upload_bytes, compresslevel=1)
        log.info("Uploading volume to server (%.1f MB%s) ...",
                 len(upload_bytes) / (1024 * 1024),
                 ", gzip" if compress_upload else "")

        self._send_json({
            "op": "load_image",
            "shape_zyx": list(self.image_shape_zyx),
            "spacing_xyz": list(self.sitk_image.GetSpacing()),
            "origin_xyz": list(self.sitk_image.GetOrigin()),
            "direction": list(self.sitk_image.GetDirection()),
            "dtype": str(arr.dtype),
            "media_id": self.config.media_id,
            "size_bytes": len(upload_bytes),
            "compression": "gzip" if compress_upload else "none",
        })
        self._send_binary(upload_bytes)

        loaded = self._recv_json()
        if loaded.get("op") != "loaded":
            raise RemoteSegmenterError(
                f"Server failed to load volume: {loaded!r}"
            )
        log.info("Volume loaded on server in %.1fs",
                 loaded.get("duration_s", 0.0))

        self.prompts_log: list[Prompt] = []

    # ------------------------------------------------------------------
    # Tiny WS helpers
    # ------------------------------------------------------------------

    def _send_json(self, obj: dict) -> None:
        try:
            self._ws.send(json.dumps(obj, separators=(",", ":")))
        except Exception as exc:
            raise RemoteSegmenterError(
                f"WS send_json failed: {exc}"
            ) from exc

    def _send_binary(self, data: bytes) -> None:
        try:
            from websocket import ABNF  # type: ignore
            self._ws.send(data, opcode=ABNF.OPCODE_BINARY)
        except Exception as exc:
            raise RemoteSegmenterError(
                f"WS send_binary failed: {exc}"
            ) from exc

    def _recv_json(self) -> dict:
        try:
            raw = self._ws.recv()
        except Exception as exc:
            raise RemoteSegmenterError(
                f"WS recv_json failed: {exc}"
            ) from exc
        if isinstance(raw, bytes):
            raise RemoteSegmenterError(
                "Expected JSON text from server, got binary frame"
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteSegmenterError(
                f"Invalid JSON from server: {raw!r}"
            ) from exc
        if data.get("op") == "error":
            raise RemoteSegmenterError(
                f"Server error: {data.get('message')} "
                f"(context: {data.get('context')})"
            )
        return data

    def _recv_binary(self, size: int) -> bytes:
        # websocket-client gives us complete frames, so a single recv
        # returns the entire binary payload sent by the server.
        try:
            raw = self._ws.recv()
        except Exception as exc:
            raise RemoteSegmenterError(
                f"WS recv_binary failed: {exc}"
            ) from exc
        if not isinstance(raw, (bytes, bytearray)):
            raise RemoteSegmenterError(
                f"Expected binary frame of {size} bytes, got text: {raw!r:.200}"
            )
        if len(raw) != size:
            raise RemoteSegmenterError(
                f"Binary frame size mismatch: got {len(raw)} bytes, "
                f"expected {size}"
            )
        return bytes(raw)

    def _apply_step_reply(self, reply: dict) -> None:
        """Read mask payload that follows a step_done envelope."""
        np = self._np
        size = int(reply.get("mask_size_bytes", 0))
        if size <= 0:
            # No mask included — leave self.mask untouched.
            return
        compression = reply.get("compression", "gzip")
        payload = self._recv_binary(size)
        if compression == "gzip":
            payload = gzip.decompress(payload)
        elif compression != "none":
            raise RemoteSegmenterError(
                f"Unknown mask compression: {compression}"
            )
        expected = self.mask.size  # uint8 -> 1 byte each
        if len(payload) != expected:
            raise RemoteSegmenterError(
                f"Mask payload size {len(payload)} != expected {expected}"
            )
        new_mask = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.image_shape_zyx
        )
        # Copy so self.mask remains writable downstream.
        self.mask = np.array(new_mask, copy=True)

    # ------------------------------------------------------------------
    # Public API — must mirror nninteractive_segment.Segmenter
    # ------------------------------------------------------------------

    @property
    def device(self):  # backward-compat: callers may print this
        return f"remote:{self.remote_device}"

    @property
    def model_path(self):
        return f"remote:{self.remote_model}"

    @property
    def mask_array(self):
        return self.mask

    def voxel_count(self) -> int:
        return int((self.mask > 0).sum())

    def volume_mm3(self) -> float:
        spacing = self.sitk_image.GetSpacing()
        return self.voxel_count() * float(spacing[0] * spacing[1] * spacing[2])

    def bounding_box(self) -> Optional[dict]:
        np = self._np
        if self.mask.sum() == 0:
            return None
        zs, ys, xs = np.where(self.mask > 0)
        return {
            "x": [int(xs.min()), int(xs.max())],
            "y": [int(ys.min()), int(ys.max())],
            "z": [int(zs.min()), int(zs.max())],
        }

    def add_point(self, x: int, y: int, z: int, *,
                  positive: bool = True, label: str = "") -> None:
        self._send_json({
            "op": "add_point",
            "x": int(x), "y": int(y), "z": int(z),
            "positive": bool(positive),
            "label": str(label or ""),
        })
        reply = self._recv_json()
        if reply.get("op") != "step_done":
            raise RemoteSegmenterError(
                f"add_point: unexpected reply {reply!r}"
            )
        self._apply_step_reply(reply)
        self.prompts_log.append(Prompt(
            kind="point", coords=(int(x), int(y), int(z)),
            include=bool(positive), label=label,
        ))

    def add_bbox(self, x_range: Iterable[int], y_range: Iterable[int],
                 z_range: Iterable[int], *,
                 positive: bool = True, label: str = "") -> None:
        bbox = [
            [int(x_range[0]), int(x_range[1])],
            [int(y_range[0]), int(y_range[1])],
            [int(z_range[0]), int(z_range[1])],
        ]
        self._send_json({
            "op": "add_bbox",
            "x": bbox[0], "y": bbox[1], "z": bbox[2],
            "positive": bool(positive),
            "label": str(label or ""),
        })
        reply = self._recv_json()
        if reply.get("op") != "step_done":
            raise RemoteSegmenterError(
                f"add_bbox: unexpected reply {reply!r}"
            )
        self._apply_step_reply(reply)
        self.prompts_log.append(Prompt(
            kind="bbox", coords=bbox, include=bool(positive), label=label,
        ))

    def reset_segment(self) -> None:
        self._send_json({"op": "reset"})
        reply = self._recv_json()
        if reply.get("op") != "step_done":
            raise RemoteSegmenterError(
                f"reset: unexpected reply {reply!r}"
            )
        self._apply_step_reply(reply)
        # Belt-and-braces: server sends an all-zero mask after reset, but
        # if compression=none and size=0 was sent we manually zero locally.
        self.mask.fill(0)
        self.prompts_log.append(Prompt(kind="reset"))

    # ------------------------------------------------------------------
    # Persistence — same files as the local Segmenter, written locally
    # using the SimpleITK geometry of the loaded volume.
    # ------------------------------------------------------------------

    def save_labelmap(self, name: str = "") -> str:
        sitk = self._sitk
        np = self._np

        out_name = name or f"{self.config.media_id}_nni_labelmap.nii.gz"
        out_path = self.output_dir / out_name
        mask = self.mask.astype(np.uint8)

        seg_image = sitk.GetImageFromArray(mask)
        seg_image.CopyInformation(self.sitk_image)
        sitk.WriteImage(seg_image, str(out_path), useCompression=True)
        log.info("Wrote labelmap: %s (%d voxels, %.2f mm^3)",
                 out_path, self.voxel_count(), self.volume_mm3())
        return str(out_path)

    def save_orthogonal_previews(self, name_prefix: str = "") -> list[str]:
        np = self._np
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed; skipping preview screenshots")
            return []

        sitk = self._sitk
        arr = sitk.GetArrayFromImage(self.sitk_image)
        mask = self.mask
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
                ax.imshow(overlay, cmap="autumn", alpha=0.45, origin="lower")
            ax.set_title(f"{view_name} (slice {s}) — {self.config.media_id}")
            ax.set_axis_off()
            out_path = self.output_dir / f"{prefix}_{view_name}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            previews.append(str(out_path))
        return previews

    def export_summary(self, extra: Optional[dict] = None) -> str:
        summary = {
            "media_id": self.config.media_id,
            "input_path": str(self.input_path),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "image_shape_zyx": list(self.image_shape_zyx),
            "voxel_spacing_xyz": list(self.sitk_image.GetSpacing()),
            "n_prompts": len(self.prompts_log),
            "prompts": [p.to_dict() for p in self.prompts_log],
            "voxel_count": self.voxel_count(),
            "volume_mm3": round(self.volume_mm3(), 3),
            "bounding_box_voxels": self.bounding_box(),
            "remote_url": self.ws_url,
        }
        if extra:
            summary.update(extra)
        out_path = self.output_dir / f"{self.config.media_id}_nni_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        return str(out_path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._send_json({"op": "close"})
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tiny CLI: ping the server, print info
# ---------------------------------------------------------------------------


def _ping(url: str) -> int:
    from websocket import create_connection  # type: ignore
    ws = create_connection(url, timeout=10)
    try:
        ws.send(json.dumps({"op": "hello", "version": PROTOCOL_VERSION}))
        ack = json.loads(ws.recv())
        ws.send(json.dumps({"op": "info"}))
        info = json.loads(ws.recv())
        ws.send(json.dumps({"op": "ping"}))
        pong = json.loads(ws.recv())
        ws.send(json.dumps({"op": "close"}))
    finally:
        try:
            ws.close()
        except Exception:
            pass
    print(json.dumps({"ack": ack, "info": info, "pong": pong},
                     indent=2, default=str))
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Ping/inspect a remote nnInteractive WS server")
    p.add_argument("--url", default=DEFAULT_WS_URL)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    return _ping(args.url)


if __name__ == "__main__":
    sys.exit(main())
