"""
Push / list / hash NIfTI volumes inside a remote 3D Slicer process.

The bright-seed orchestrator needs to load a freshly cropped CT into the
remote Slicer instance before invoking nnInteractive on it. Slicer's
built-in Web Server module exposes ``/slicer/exec`` which accepts an
arbitrary Python source string and evaluates it inside the Slicer
process; that's the only side-channel we have to drive Slicer remotely.

The simplest workable file-transfer scheme over ``/slicer/exec`` is to
inline the volume's bytes as base64 inside the Python source we POST.
Round-trip overhead is acceptable for the post-crop volumes used in this
pilot (~50-200 MB each, ~67-265 MB after base64). Larger volumes should
use a different transport (e.g. a server-side download recipe).

Functions
---------
- ``push_volume(base_url, nifti_path, name=None)``: read *nifti_path*,
  base64-encode it, POST a recipe that decodes + writes a temp file +
  loads it via ``slicer.util.loadVolume``, and sets it active. Returns
  the parsed JSON reply.
- ``list_volumes(base_url)``: list current ``vtkMRMLScalarVolumeNode``s.
- ``set_active_volume(base_url, name)``: select the volume by name.

Constants
---------
- ``LOAD_NIFTI_BASE64_SRC_TEMPLATE``: format with ``data_b64`` and
  ``name``.
- ``LIST_VOLUMES_SRC``: no formatting required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# HTTP helper (mirrors the one in slicer_remote_bright_seed)
# ---------------------------------------------------------------------------

def _post_python(base_url: str, source: str, timeout: float = 600) -> dict:
    body = source.encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/slicer/exec",
        data=body, method="POST",
        headers={"Content-Type": "text/plain"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        content = e.read()
        status = e.code
    if status != 200:
        raise RuntimeError(
            f"/slicer/exec -> HTTP {status}: {content[:300]!r}"
        )
    try:
        result = json.loads(content)
    except Exception:
        raise RuntimeError(f"non-JSON exec reply: {content[:300]!r}")
    result["_dt_s"] = round(time.time() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Slicer-side recipes (run via /slicer/exec)
# ---------------------------------------------------------------------------

# We embed the base64 directly via str.format so the Slicer source is
# self-contained. The placeholder uses {data_b64} (and {name},
# {sha256_expected}). Use ``LOAD_NIFTI_BASE64_SRC_TEMPLATE.format(...)``.
#
# Notes:
# - The remote temp file is keyed on the expected sha256 so re-pushes of
#   the same content are idempotent: on a hash match, we reuse the
#   existing file.
# - We delete any previously-loaded volume node with the same name so we
#   don't pile up stale copies across runs.
# - We verify the decoded bytes' sha256 against ``sha256_expected`` and
#   surface a mismatch as ``status: "sha256_mismatch"`` instead of
#   silently loading bad data.
# --------------------------------------------------------------------
# Chunked-upload recipes
# --------------------------------------------------------------------
# We can't push a 50-300 MB cropped CT through the Exosphere proxy in a
# single /slicer/exec POST: even after gzip+base64 compresses well, the
# nginx proxy fronting the Slicer Web Server has a per-request idle
# timeout (~60 s) shorter than `slicer.util.loadVolume` on a fresh
# NIfTI. The workaround is to land the bytes on the Jetstream
# filesystem in small chunks first, then load it as a fast separate
# call.
#
# Three short recipes (each a single fast POST):
#   1. INIT_UPLOAD_SRC: create the destination temp file (truncate).
#   2. APPEND_UPLOAD_SRC: base64-decode this chunk + append to the file.
#   3. LOAD_FROM_PATH_SRC: verify sha256, then loadVolume from the
#      already-on-disk file and set it active.
#
# Every recipe injects parameters via a tiny prelude rather than
# str.format() so the body can use {dict} literals freely.

_INIT_UPLOAD_BODY = """\
import os, tempfile, traceback
out = {}
try:
    td = os.path.join(tempfile.gettempdir(), "ms_remote_volumes")
    os.makedirs(td, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in _NAME)
    path = os.path.join(td, _SHA_EXPECTED[:16] + "_" + safe + ".nii.gz")
    open(path, "wb").close()  # truncate / create
    out["status"] = "ok"
    out["remote_path"] = path
    out["jetstream_dir"] = td
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""

_APPEND_UPLOAD_BODY = """\
import base64, os, hashlib, traceback
out = {}
try:
    data = base64.b64decode(_DATA_B64)
    with open(_PATH, "ab") as f:
        f.write(data)
    out["status"] = "ok"
    out["chunk_size"] = len(data)
    out["chunk_sha256"] = hashlib.sha256(data).hexdigest()
    out["file_size_after"] = os.path.getsize(_PATH)
    out["chunk_index"] = _CHUNK_INDEX
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""

_LOAD_FROM_PATH_BODY = """\
import slicer, hashlib, os, traceback
out = {}
try:
    if not os.path.exists(_PATH):
        out["status"] = "missing"
    else:
        size = os.path.getsize(_PATH)
        out["size_bytes"] = size
        h = hashlib.sha256()
        with open(_PATH, "rb") as f:
            while True:
                blk = f.read(1 << 20)
                if not blk:
                    break
                h.update(blk)
        sha = h.hexdigest()
        out["sha256"] = sha
        out["remote_path"] = _PATH
        if _SHA_EXPECTED and sha != _SHA_EXPECTED:
            out["status"] = "sha256_mismatch"
            out["sha256_expected"] = _SHA_EXPECTED
        else:
            for existing in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                if existing.GetName() == _NAME:
                    slicer.mrmlScene.RemoveNode(existing)
            node = slicer.util.loadVolume(_PATH, properties={"name": _NAME})
            if node is None:
                out["status"] = "load_failed"
            else:
                sel = slicer.app.applicationLogic().GetSelectionNode()
                sel.SetActiveVolumeID(node.GetID())
                slicer.app.applicationLogic().PropagateVolumeSelection(0)
                slicer.util.setSliceViewerLayers(background=node, fit=True)
                arr = slicer.util.arrayFromVolume(node)
                out["status"] = "ok"
                out["volume_id"] = node.GetID()
                out["volume_name"] = node.GetName()
                out["shape_kji"] = list(arr.shape)
                out["dtype"] = str(arr.dtype)
                out["spacing_mm"] = [round(s, 6) for s in node.GetSpacing()]
                out["origin"] = [round(o, 6) for o in node.GetOrigin()]
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


def _build_init_source(name: str, sha256_expected: str) -> str:
    return (
        f"_NAME = {name!r}\n"
        f"_SHA_EXPECTED = {sha256_expected!r}\n"
        + _INIT_UPLOAD_BODY
    )


def _build_append_source(path: str, data_b64: str, chunk_index: int) -> str:
    return (
        f"_PATH = {path!r}\n"
        f"_DATA_B64 = {data_b64!r}\n"
        f"_CHUNK_INDEX = {int(chunk_index)}\n"
        + _APPEND_UPLOAD_BODY
    )


def _build_load_from_path_source(path: str, name: str,
                                   sha256_expected: str) -> str:
    return (
        f"_PATH = {path!r}\n"
        f"_NAME = {name!r}\n"
        f"_SHA_EXPECTED = {sha256_expected!r}\n"
        + _LOAD_FROM_PATH_BODY
    )


# Single-shot loader (kept for callers that already have the file on
# the remote box and just want it loaded). The orchestrator no longer
# uses this path for fresh uploads -- it goes through the chunked path.
LOAD_NIFTI_BASE64_SRC_BODY = """\
import slicer, base64, hashlib, os, tempfile, traceback
out = {}
try:
    data = base64.b64decode(_DATA_B64)
    sha = hashlib.sha256(data).hexdigest()
    out["sha256"] = sha
    out["size_bytes"] = len(data)
    if _SHA_EXPECTED and sha != _SHA_EXPECTED:
        out["status"] = "sha256_mismatch"
        out["sha256_expected"] = _SHA_EXPECTED
    else:
        td = os.path.join(tempfile.gettempdir(), "ms_remote_volumes")
        os.makedirs(td, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in _NAME)
        path = os.path.join(td, sha[:16] + "_" + safe + ".nii.gz")
        if not os.path.exists(path) or os.path.getsize(path) != len(data):
            with open(path, "wb") as f:
                f.write(data)
        out["remote_path"] = path
        for existing in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if existing.GetName() == _NAME:
                slicer.mrmlScene.RemoveNode(existing)
        node = slicer.util.loadVolume(path, properties={"name": _NAME})
        if node is None:
            out["status"] = "load_failed"
        else:
            sel = slicer.app.applicationLogic().GetSelectionNode()
            sel.SetActiveVolumeID(node.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection(0)
            slicer.util.setSliceViewerLayers(background=node, fit=True)
            arr = slicer.util.arrayFromVolume(node)
            out["status"] = "ok"
            out["volume_id"] = node.GetID()
            out["volume_name"] = node.GetName()
            out["shape_kji"] = list(arr.shape)
            out["dtype"] = str(arr.dtype)
            out["spacing_mm"] = [round(s, 6) for s in node.GetSpacing()]
            out["origin"] = [round(o, 6) for o in node.GetOrigin()]
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


def _build_load_source(data_b64: str, name: str,
                       sha256_expected: str) -> str:
    """Compose the full /slicer/exec source for push_volume.

    We keep ``LOAD_NIFTI_BASE64_SRC_BODY`` parameter-free and prepend a
    tiny prelude that defines the inputs as plain assignments. This
    avoids the brace-escaping headache of formatting a Python source
    string that contains its own dict literals.
    """
    prelude = (
        f"_DATA_B64 = {data_b64!r}\n"
        f"_NAME = {name!r}\n"
        f"_SHA_EXPECTED = {sha256_expected!r}\n"
    )
    return prelude + LOAD_NIFTI_BASE64_SRC_BODY


def LOAD_NIFTI_BASE64_SRC_TEMPLATE(data_b64: str, name: str,
                                   sha256_expected: str = "") -> str:
    """Function-style accessor mimicking a format-string entry point."""
    return _build_load_source(data_b64, name, sha256_expected)


# Lightweight scene listing for sanity / dedup.
LIST_VOLUMES_SRC = """\
import slicer, traceback
out = {"volumes": [], "active_id": None}
try:
    sel = slicer.app.applicationLogic().GetSelectionNode()
    out["active_id"] = sel.GetActiveVolumeID() if sel else None
    for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        try:
            arr = slicer.util.arrayFromVolume(v)
            shape = list(arr.shape)
        except Exception:
            shape = None
        out["volumes"].append({
            "id": v.GetID(),
            "name": v.GetName(),
            "shape_kji": shape,
            "spacing_mm": [round(s, 6) for s in v.GetSpacing()],
            "is_active": (sel is not None and v.GetID() == sel.GetActiveVolumeID()),
        })
    out["status"] = "ok"
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


# Set the active volume by *name*. Mirrors the helper in
# slicer_remote_bright_seed but exposed as a reusable recipe.
_SET_ACTIVE_VOLUME_SRC_BODY = """\
import slicer, traceback
out = {}
try:
    found = None
    for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if v.GetName() == _TARGET:
            found = v
            break
    if found is None:
        out["status"] = "not_found"
        out["available"] = [v.GetName() for v in
                             slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")]
    else:
        sel = slicer.app.applicationLogic().GetSelectionNode()
        sel.SetActiveVolumeID(found.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection(0)
        slicer.util.setSliceViewerLayers(background=found, fit=True)
        out["status"] = "ok"
        out["volume_id"] = found.GetID()
        out["volume_name"] = found.GetName()
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


def SET_ACTIVE_VOLUME_SRC_TEMPLATE(name: str) -> str:
    """Build the /slicer/exec source for set_active_volume."""
    return f"_TARGET = {name!r}\n" + _SET_ACTIVE_VOLUME_SRC_BODY


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def push_volume(base_url: str, nifti_path: Path,
                name: Optional[str] = None,
                chunk_bytes: int = 6 * 1024 * 1024,
                per_chunk_timeout: float = 60.0,
                load_timeout: float = 240.0,
                progress: Optional[Any] = None) -> dict:
    """Upload *nifti_path* to the Jetstream filesystem in chunks, then load.

    The chunked path keeps every individual ``/slicer/exec`` call below
    the ~60 s nginx idle timeout that fronts the Slicer Web Server on
    Exosphere. Each chunk decodes a base64-encoded slice of the file
    and **appends** to the same temp file on Jetstream's local disk;
    once the upload is complete, a separate /exec call invokes
    ``slicer.util.loadVolume`` on the on-disk path (no upload, just a
    file open).

    Parameters
    ----------
    base_url : str
        Slicer Web Server URL (https://...-2016.proxy-js2-iu.exosphere.app/).
    nifti_path : Path
        Local NIfTI file to push.
    name : str, optional
        Slicer scene node name. Defaults to ``nifti_path.stem``.
    chunk_bytes : int
        Size of each upload chunk in raw bytes (default 6 MiB,
        ~8 MiB after base64). 6 MiB on a typical home connection
        uploads in 5–15 s, well under the proxy idle timeout.
    per_chunk_timeout : float
        Per-chunk POST timeout (default 60 s).
    load_timeout : float
        Final loadVolume call timeout (default 240 s; loadVolume on a
        50 MB NIfTI typically takes 20–60 s).
    progress : callable(int, int), optional
        ``progress(uploaded_bytes, total_bytes)`` callback for
        long-running upload feedback.

    Returns
    -------
    dict
        Reply of the loadVolume step, augmented with ``local_sha256``,
        ``local_size_bytes``, ``local_path``, ``remote_path``, and
        per-stage timing info.
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(nifti_path)
    data = nifti_path.read_bytes()
    total = len(data)
    sha = hashlib.sha256(data).hexdigest()
    if name is None:
        name = nifti_path.stem  # "ct_cropped.nii" for "ct_cropped.nii.gz"

    timings: dict = {"chunks": [], "init_s": None, "load_s": None}
    init = _post_python(
        base_url,
        _build_init_source(name=name, sha256_expected=sha),
        timeout=per_chunk_timeout,
    )
    timings["init_s"] = init.get("_dt_s")
    if init.get("status") != "ok":
        return {**init, "stage": "init",
                 "local_sha256": sha, "local_size_bytes": total,
                 "local_path": str(nifti_path), "_timings": timings}
    remote_path = init["remote_path"]

    n_chunks = (total + chunk_bytes - 1) // max(1, chunk_bytes)
    sent = 0
    for i in range(n_chunks):
        chunk = data[i * chunk_bytes:(i + 1) * chunk_bytes]
        b64 = base64.b64encode(chunk).decode("ascii")
        reply = _post_python(
            base_url,
            _build_append_source(path=remote_path, data_b64=b64,
                                   chunk_index=i),
            timeout=per_chunk_timeout,
        )
        timings["chunks"].append({
            "i": i, "bytes": len(chunk),
            "dt_s": reply.get("_dt_s"),
            "remote_size_after": reply.get("file_size_after"),
            "status": reply.get("status"),
        })
        if reply.get("status") != "ok":
            return {**reply, "stage": "append", "chunk_index": i,
                     "local_sha256": sha, "local_size_bytes": total,
                     "local_path": str(nifti_path),
                     "remote_path": remote_path, "_timings": timings}
        sent += len(chunk)
        if progress is not None:
            try:
                progress(sent, total)
            except Exception:
                pass

    final = _post_python(
        base_url,
        _build_load_from_path_source(path=remote_path, name=name,
                                        sha256_expected=sha),
        timeout=load_timeout,
    )
    timings["load_s"] = final.get("_dt_s")
    final["local_sha256"] = sha
    final["local_size_bytes"] = total
    final["local_path"] = str(nifti_path)
    final["remote_path"] = remote_path
    final["_timings"] = timings
    final["chunk_bytes"] = chunk_bytes
    final["n_chunks"] = n_chunks
    return final


def list_volumes(base_url: str, timeout: float = 30) -> dict:
    return _post_python(base_url, LIST_VOLUMES_SRC, timeout=timeout)


def set_active_volume(base_url: str, name: str, timeout: float = 30) -> dict:
    return _post_python(
        base_url,
        SET_ACTIVE_VOLUME_SRC_TEMPLATE(name),
        timeout=timeout,
    )


__all__ = [
    "LOAD_NIFTI_BASE64_SRC_TEMPLATE",
    "LIST_VOLUMES_SRC",
    "SET_ACTIVE_VOLUME_SRC_TEMPLATE",
    "push_volume",
    "list_volumes",
    "set_active_volume",
]
