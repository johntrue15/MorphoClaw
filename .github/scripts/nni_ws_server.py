"""
WebSocket server that exposes nnInteractive over the wire.

Designed to run on a beefy box (e.g. a Jetstream2 / MorphoCloud VM) so a
local orchestrator on a memory-constrained machine (e.g. 16 GB Mac mini)
can drive an LLM-in-the-loop nnInteractive paint session without holding
the model in local RAM.

Usage
-----
    python nni_ws_server.py --host 0.0.0.0 --port 8765

Or, when reachable only via SSH tunnel:

    python nni_ws_server.py --host 127.0.0.1 --port 8765
    # local:  ssh -L 8765:127.0.0.1:8765 user@remote-host
    # local:  NNI_REMOTE_WS=ws://localhost:8765 ./test_chameleon_stapes.sh

The protocol is documented in ``nninteractive_remote.py``.

This script depends on the same packages as the local Segmenter
(``nninteractive_segment.Segmenter``) plus the ``websockets`` library.
Install it (alongside nnInteractive) via::

    pip install websockets>=12

Security
--------
The server has no authentication; it is intended to live behind an SSH
tunnel (default ``--host 127.0.0.1``) and serve a single trusted client.
Do not bind to a public interface without a reverse proxy that handles
authentication and TLS.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Defer heavy imports (numpy, torch, nnInteractive) until first use.

log = logging.getLogger("nni_ws_server")

PROTOCOL_VERSION = 1

# Reuse the local Segmenter as the inference backend. We import it lazily
# because importing it pulls torch / nnInteractive which is slow.
SEGMENTER = None       # type: Optional[object]
SEGMENTER_NP = None    # numpy module reference
SEGMENTER_LOCK = asyncio.Lock()
INSTALL_HINT = (
    "Run install_nninteractive_remote.sh on this host to set up the venv "
    "and prefetch the nnInteractive_v1.0 weights."
)


def _segmenter_module():
    """Lazy import to keep startup fast and surface clear errors."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from nninteractive_segment import (   # noqa: WPS433
        NNInteractiveUnavailable, Segmenter, SegmenterConfig, _select_device,
    )
    import torch
    import numpy as np
    return {
        "NNInteractiveUnavailable": NNInteractiveUnavailable,
        "Segmenter": Segmenter,
        "SegmenterConfig": SegmenterConfig,
        "torch": torch,
        "np": np,
    }


# ---------------------------------------------------------------------------
# WebSocket helpers (websockets v12+ API: send(text|bytes), recv() -> str|bytes)
# ---------------------------------------------------------------------------


async def _send_json(ws, obj: dict) -> None:
    await ws.send(json.dumps(obj, separators=(",", ":")))


async def _send_error(ws, message: str, context: str = "") -> None:
    await _send_json(ws, {
        "op": "error",
        "message": message,
        "context": context,
    })


async def _recv_json(ws) -> dict:
    raw = await ws.recv()
    if isinstance(raw, bytes):
        raise RuntimeError("Expected JSON text frame, got binary")
    return json.loads(raw)


async def _recv_binary(ws, expected_size: int) -> bytes:
    raw = await ws.recv()
    if not isinstance(raw, (bytes, bytearray)):
        raise RuntimeError(
            f"Expected binary frame of {expected_size} bytes, got text"
        )
    if expected_size and len(raw) != expected_size:
        raise RuntimeError(
            f"Binary frame size mismatch: got {len(raw)} bytes, "
            f"expected {expected_size}"
        )
    return bytes(raw)


# ---------------------------------------------------------------------------
# Per-session handlers
# ---------------------------------------------------------------------------


class Session:
    """Holds one Segmenter + its volume metadata for a single client."""

    def __init__(self, mods: dict):
        self.mods = mods
        self.segmenter = None  # type: Optional[object]
        self.shape_zyx = None
        self.spacing_xyz = None
        self.dtype = None
        self.media_id = "remote_session"

    async def handle_load_image(self, ws, msg: dict) -> None:
        """Receive the volume bytes, write to a temp NIfTI, init Segmenter."""
        np = self.mods["np"]
        Segmenter = self.mods["Segmenter"]
        SegmenterConfig = self.mods["SegmenterConfig"]
        import SimpleITK as sitk

        shape_zyx = tuple(int(v) for v in msg.get("shape_zyx", []))
        spacing_xyz = tuple(float(v) for v in msg.get("spacing_xyz", []))
        origin_xyz = tuple(float(v) for v in msg.get("origin_xyz", [0, 0, 0]))
        direction = tuple(float(v) for v in msg.get(
            "direction", [1, 0, 0, 0, 1, 0, 0, 0, 1]
        ))
        dtype_str = str(msg.get("dtype", "float32"))
        media_id = str(msg.get("media_id", "remote_session"))
        size_bytes = int(msg.get("size_bytes", 0))
        compression = str(msg.get("compression", "none"))

        if len(shape_zyx) != 3:
            await _send_error(ws, f"shape_zyx must be 3D, got {shape_zyx}",
                              context="load_image")
            return

        log.info("Receiving volume %s (%s, %.1f MB, comp=%s)",
                 shape_zyx, dtype_str, size_bytes / (1024 * 1024), compression)

        payload = await _recv_binary(ws, size_bytes)
        if compression == "gzip":
            payload = gzip.decompress(payload)
        elif compression != "none":
            await _send_error(ws, f"Unknown compression: {compression}",
                              context="load_image")
            return

        try:
            arr = np.frombuffer(payload, dtype=np.dtype(dtype_str)).reshape(shape_zyx)
        except ValueError as exc:
            await _send_error(ws, f"Failed to reshape payload: {exc}",
                              context="load_image")
            return

        # Persist as a transient NIfTI so we can reuse the existing
        # Segmenter ctor unchanged. The file lives in /tmp and is cleared
        # on the next load_image.
        tmpdir = Path(os.environ.get(
            "NNI_REMOTE_TMP_DIR", "/tmp/nni_ws_session"
        ))
        tmpdir.mkdir(parents=True, exist_ok=True)
        nifti_path = tmpdir / f"{media_id}.nii.gz"
        log.info("Writing temp NIfTI: %s", nifti_path)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing_xyz)
        img.SetOrigin(origin_xyz)
        img.SetDirection(direction)
        sitk.WriteImage(img, str(nifti_path), useCompression=True)

        # Build the segmenter (this loads the model on first call; we
        # could share it between sessions but keeping it per-session is
        # safer and not significantly more expensive on a warm box).
        cfg = SegmenterConfig(
            input_path=str(nifti_path),
            output_dir=str(tmpdir / "outputs"),
            media_id=media_id,
        )
        log.info("Constructing nnInteractive Segmenter (this loads the model)...")
        t0 = time.time()
        try:
            self.segmenter = Segmenter(cfg)
        except Exception as exc:
            log.exception("Segmenter init failed")
            await _send_error(ws, f"Segmenter init failed: {exc}",
                              context="load_image")
            return
        duration = time.time() - t0

        self.shape_zyx = shape_zyx
        self.spacing_xyz = spacing_xyz
        self.dtype = dtype_str
        self.media_id = media_id

        await _send_json(ws, {
            "op": "loaded",
            "shape_zyx": list(shape_zyx),
            "spacing_xyz": list(spacing_xyz),
            "duration_s": round(duration, 2),
        })

    async def _send_step_done(self, ws, t0: float) -> None:
        """Send a step_done envelope + binary mask payload."""
        np = self.mods["np"]
        seg = self.segmenter
        mask = seg.mask_array.astype(np.uint8)
        raw = mask.tobytes()
        compressed = gzip.compress(raw, compresslevel=1)

        voxel_count = int((mask > 0).sum())
        await _send_json(ws, {
            "op": "step_done",
            "voxel_count": voxel_count,
            "duration_s": round(time.time() - t0, 2),
            "mask_size_bytes": len(compressed),
            "compression": "gzip",
        })
        await ws.send(compressed)

    async def handle_add_point(self, ws, msg: dict) -> None:
        if self.segmenter is None:
            await _send_error(ws, "load_image first", context="add_point")
            return
        t0 = time.time()
        try:
            self.segmenter.add_point(
                int(msg["x"]), int(msg["y"]), int(msg["z"]),
                positive=bool(msg.get("positive", True)),
                label=str(msg.get("label", "")),
            )
        except Exception as exc:
            log.exception("add_point failed")
            await _send_error(ws, str(exc), context="add_point")
            return
        await self._send_step_done(ws, t0)

    async def handle_add_bbox(self, ws, msg: dict) -> None:
        if self.segmenter is None:
            await _send_error(ws, "load_image first", context="add_bbox")
            return
        t0 = time.time()
        try:
            self.segmenter.add_bbox(
                msg["x"], msg["y"], msg["z"],
                positive=bool(msg.get("positive", True)),
                label=str(msg.get("label", "")),
            )
        except Exception as exc:
            log.exception("add_bbox failed")
            await _send_error(ws, str(exc), context="add_bbox")
            return
        await self._send_step_done(ws, t0)

    async def handle_reset(self, ws) -> None:
        if self.segmenter is None:
            await _send_error(ws, "load_image first", context="reset")
            return
        t0 = time.time()
        try:
            self.segmenter.reset_segment()
        except Exception as exc:
            log.exception("reset failed")
            await _send_error(ws, str(exc), context="reset")
            return
        await self._send_step_done(ws, t0)

    async def handle_get_mask(self, ws) -> None:
        if self.segmenter is None:
            await _send_error(ws, "load_image first", context="get_mask")
            return
        await self._send_step_done(ws, time.time())


# ---------------------------------------------------------------------------
# Top-level connection handler
# ---------------------------------------------------------------------------


async def _handle_client(ws, mods: dict) -> None:
    log.info("Client connected: %s",
             getattr(ws, "remote_address", "unknown"))
    sess = Session(mods)
    try:
        first = await _recv_json(ws)
        if first.get("op") != "hello":
            await _send_error(ws, f"Expected hello, got {first.get('op')}",
                              context="handshake")
            return
        if first.get("version") != PROTOCOL_VERSION:
            await _send_error(ws,
                              f"Protocol version mismatch (server={PROTOCOL_VERSION}, "
                              f"client={first.get('version')})",
                              context="handshake")
            return

        # Optional shared-secret auth. If NNI_WS_TOKEN is set on the
        # server side, every client must send the same value in its
        # "hello" frame. The Exosphere proxy is publicly reachable, so
        # we strongly recommend setting this in production deployments.
        expected_token = os.environ.get("NNI_WS_TOKEN", "").strip()
        if expected_token:
            client_token = str(first.get("token", "")).strip()
            # constant-time comparison to avoid trivial timing oracles
            import hmac
            ok = hmac.compare_digest(expected_token, client_token)
            if not ok:
                log.warning("Rejecting client %s: bad/missing NNI_WS_TOKEN",
                            getattr(ws, "remote_address", "?"))
                await _send_error(ws, "auth failed (bad NNI_WS_TOKEN)",
                                  context="handshake")
                return
        torch = mods["torch"]
        # Probe device cheaply (don't actually allocate)
        if torch.cuda.is_available():
            device_str = (
                f"cuda:0 ({torch.cuda.get_device_name(0)})"
            )
        elif (getattr(torch.backends, "mps", None)
              and torch.backends.mps.is_available()):
            device_str = "mps"
        else:
            device_str = "cpu"

        await _send_json(ws, {
            "op": "hello_ack",
            "version": PROTOCOL_VERSION,
            "device": device_str,
            "model": os.environ.get("NNINTERACTIVE_MODEL",
                                    "nnInteractive_v1.0"),
            "model_loaded": False,
        })

        while True:
            try:
                msg = await _recv_json(ws)
            except Exception as exc:
                log.warning("recv_json failed: %s", exc)
                break
            op = msg.get("op")
            if op == "load_image":
                await sess.handle_load_image(ws, msg)
            elif op == "add_point":
                await sess.handle_add_point(ws, msg)
            elif op == "add_bbox":
                await sess.handle_add_bbox(ws, msg)
            elif op == "reset":
                await sess.handle_reset(ws)
            elif op == "get_mask":
                await sess.handle_get_mask(ws)
            elif op == "info":
                info = {
                    "op": "info_reply",
                    "device": device_str,
                    "shape_zyx": list(sess.shape_zyx) if sess.shape_zyx else None,
                    "media_id": sess.media_id,
                    "torch_version": str(torch.__version__),
                }
                if torch.cuda.is_available():
                    info["cuda_mem_total_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory
                        / (1024 ** 3), 2,
                    )
                await _send_json(ws, info)
            elif op == "ping":
                await _send_json(ws, {"op": "pong", "ts": time.time()})
            elif op == "close":
                log.info("Client requested close")
                break
            else:
                await _send_error(ws, f"Unknown op: {op}",
                                  context="dispatch")
    except Exception as exc:
        log.exception("client handler errored")
        try:
            await _send_error(ws, str(exc), context="handler")
        except Exception:
            pass
    finally:
        log.info("Client disconnected")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _serve(host: str, port: int) -> None:
    try:
        import websockets  # type: ignore
    except ImportError:
        log.error("Missing dependency: websockets. Install with: "
                  "pip install websockets>=12")
        sys.exit(2)

    log.info("Importing nnInteractive backend (this may take a few seconds)…")
    try:
        mods = _segmenter_module()
    except Exception:
        log.exception("Failed to import nnInteractive backend. %s", INSTALL_HINT)
        sys.exit(2)

    log.info("Listening on ws://%s:%d", host, port)

    async def handler(ws):
        await _handle_client(ws, mods)

    # Allow large frames (uncompressed volume can be >1 GB on big CTs).
    max_frame = int(os.environ.get(
        "NNI_REMOTE_MAX_FRAME_BYTES", str(2 * 1024 ** 3)  # 2 GiB
    ))
    async with websockets.serve(
        handler,
        host=host,
        port=port,
        max_size=max_frame,
        ping_interval=30,
        ping_timeout=300,
    ):
        await asyncio.Future()  # run forever


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default bind to 0.0.0.0 because in the Exosphere/JS2 deployment the
    # proxy reaches us via the box's external interface; binding only to
    # 127.0.0.1 makes us unreachable. Override with NNI_REMOTE_HOST=127.0.0.1
    # if the server is meant to be local-only (e.g. behind an SSH tunnel).
    p.add_argument("--host", default=os.environ.get("NNI_REMOTE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("NNI_REMOTE_PORT", "8765")))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        log.info("Interrupted, exiting cleanly.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
