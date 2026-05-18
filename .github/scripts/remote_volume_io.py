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
from typing import Optional


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
# Static recipe (no formatting). The caller plants the parameters as
# globals via a tiny prelude built by ``_build_load_source`` so we don't
# have to escape braces from inside an f-string template.
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


# Backwards-compatible alias for any caller that wants a single template
# they can format. ``data_b64`` / ``name`` / ``sha256_expected`` are
# substituted via repr (str.format with !r) and the body is appended.
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
                timeout: float = 600) -> dict:
    """Read *nifti_path*, base64 it, and load it inside remote Slicer.

    Returns the parsed reply from /slicer/exec, augmented with
    ``local_sha256`` and ``local_size_bytes``. On success, the volume
    is also set active in the remote scene.
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(nifti_path)
    data = nifti_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if name is None:
        name = nifti_path.stem  # "ct_cropped.nii" for "ct_cropped.nii.gz"
    b64 = base64.b64encode(data).decode("ascii")
    src = _build_load_source(data_b64=b64, name=name, sha256_expected=sha)
    reply = _post_python(base_url, src, timeout=timeout)
    reply["local_sha256"] = sha
    reply["local_size_bytes"] = len(data)
    reply["local_path"] = str(nifti_path)
    return reply


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
