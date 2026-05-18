"""
Provenance + telemetry helpers for Slicer remote experiments.

The goal is paper-grade reproducibility: a self-describing run directory
that someone reading the publication can clone, replay, and verify byte
hashes against. Every script that drives a remote Slicer instance should
use ``RunLogger`` so we get a uniform layout:

  runs/<run_id>/
    manifest.json         Immutable run metadata (run_id, args, started_utc).
    environment.json      Local + remote environment snapshot (versions,
                          device, model checkpoint hash, ...).
    inputs.json           Volume identity (name, shape, spacing, dtype,
                          content sha256). Whoever has these bytes can
                          verify they're feeding the same data.
    events.jsonl          Append-only structured event log. One JSON
                          object per line, wall-clock + elapsed_s on
                          every record.
    log.txt               Human-readable mirror of stdout (timestamped).
    artifacts/            Final outputs (per-segment NIfTI, composite
                          labelmap NIfTI). Each registered with sha256.
    artifacts/index.json  Manifest of artifacts with sha256 + bytes.
    per_step/step_NN/     Per-step screenshots + state.json (existing).
    summary.json          Final aggregate (existing).
    replay.sh             One-line reproduction (env + command).
    stop_reason.json      Machine-readable stop reason.

Most importantly:
  * Events.jsonl is append-only and crash-safe.
  * Every artifact written via ``RunLogger.write_artifact`` is hashed
    and registered in ``artifacts/index.json``.
  * Server-side telemetry recipes (volume hash, environment capture,
    segmentation export) are kept here so they're reused identically
    across scripts.
"""

from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

UTC = datetime.timezone.utc


# ---------------------------------------------------------------------------
# Local-side helpers
# ---------------------------------------------------------------------------

def _git(args: Iterable[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _detect_repo_root(start: Optional[Path] = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return here.parent


def capture_local_env(repo_root: Optional[Path] = None) -> dict:
    """Snapshot of the machine driving the experiment."""
    repo_root = repo_root or _detect_repo_root()
    return {
        "captured_utc": datetime.datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "git_repo_root": str(repo_root),
        "git_commit": _git(["rev-parse", "HEAD"], cwd=repo_root),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "git_dirty": bool(_git(["status", "--porcelain"], cwd=repo_root)),
        "git_remote": _git(["config", "--get", "remote.origin.url"], cwd=repo_root),
        "package_versions": _capture_pkg_versions([
            "numpy", "Pillow", "requests", "openai",
        ]),
        "env_vars_relevant": {
            k: ("***" if "TOKEN" in k or "KEY" in k or "SECRET" in k or "PASS" in k
                else os.environ[k])
            for k in sorted(os.environ)
            if any(p in k for p in (
                "SLICER", "NNI", "JETSTREAM", "OPENAI", "GITHUB",
                "PYTORCH", "CUDA",
            ))
        },
    }


def _capture_pkg_versions(names: Iterable[str]) -> dict:
    out = {}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as e:
            out[name] = f"<missing: {e!r}>"
    return out


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Remote (Slicer-side) telemetry recipes
# ---------------------------------------------------------------------------
# Each recipe is a string of Python that the caller posts to /slicer/exec.
# All recipes:
#   1. populate __execResult with their findings, and
#   2. wrap their work in try/except so a recoverable failure surfaces
#      as ``status: "exception"`` instead of HTTP 500.

CAPTURE_REMOTE_ENV_SRC = """\
import slicer, sys, os, traceback, hashlib, json
out = {"captured_utc": __import__('datetime').datetime.utcnow().isoformat() + 'Z'}
def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        return f"<error: {e!r}>"
try:
    out["host"] = _safe(__import__('socket').gethostname)
    out["platform"] = _safe(__import__('platform').platform)
    out["machine"] = _safe(__import__('platform').machine)
    out["python_version"] = sys.version.split()[0]
    out["python_executable"] = sys.executable
    out["slicer_version"] = slicer.app.applicationVersion
    try:
        out["slicer_revision"] = slicer.app.repositoryRevision
    except Exception:
        pass
    try:
        modules = slicer.app.moduleManager().modulesNames()
        out["slicer_module_count"] = len(modules)
    except Exception:
        pass
    # SlicerNNInteractive plugin version / source location
    fastapi_server = None
    try:
        mod = slicer.modules.slicernninteractive
        rep = mod.widgetRepresentation()
        plugin = rep.self()
        plugin_path = sys.modules[plugin.__class__.__module__].__file__
        out["slicernninteractive_module_path"] = plugin_path
        # Try to read package dir's git commit if it's a checkout
        plugin_dir = os.path.dirname(plugin_path)
        for up in range(4):
            git_dir = os.path.join(plugin_dir, ".git")
            if os.path.isdir(git_dir):
                head = open(os.path.join(git_dir, "HEAD")).read().strip()
                out["slicernninteractive_git_HEAD"] = head
                if head.startswith("ref:"):
                    ref = head.split(" ", 1)[1].strip()
                    try:
                        out["slicernninteractive_git_commit"] = open(
                            os.path.join(git_dir, ref)).read().strip()
                    except Exception:
                        pass
                break
            plugin_dir = os.path.dirname(plugin_dir)
        fastapi_server = getattr(plugin, "server", None)
        out["slicernninteractive_server_url"] = fastapi_server
    except Exception as e:
        out["slicernninteractive_err"] = repr(e)
    # The actual nnInteractive inference + torch live INSIDE the
    # SlicerNNInteractive FastAPI server (a separate Python process,
    # typically on port 1527), not in Slicer's bundled Python.
    #
    # Three-pronged capture so the paper has full provenance even
    # though the FastAPI doesn't expose any /info endpoint:
    #   1. Probe a few likely diagnostic endpoints (cheap, often 404).
    #   2. Find the PID listening on the FastAPI port via lsof / ss,
    #      then read /proc/<pid>/exe to get the Python interpreter path.
    #   3. Run that interpreter's pip + sys.version to capture the
    #      torch/nnInteractive versions exactly.
    if fastapi_server:
        try:
            import urllib.request
            for path in ("/info", "/version", "/health", "/openapi.json"):
                try:
                    req = urllib.request.Request(fastapi_server.rstrip("/") + path)
                    with urllib.request.urlopen(req, timeout=5) as r:
                        body = r.read(8 * 1024)
                        out[f"fastapi{path}_status"] = r.status
                        try:
                            out[f"fastapi{path}_body"] = body.decode("utf-8", "replace")[:4000]
                        except Exception:
                            out[f"fastapi{path}_body"] = repr(body[:200])
                except Exception as e:
                    out[f"fastapi{path}_err"] = repr(e)
        except Exception as e:
            out["fastapi_probe_err"] = repr(e)
        # Find the FastAPI process via the URL's port (default 1527),
        # then introspect its Python venv. Best-effort; failures are
        # non-fatal but recorded.
        try:
            import re, subprocess, urllib.parse
            parsed = urllib.parse.urlparse(fastapi_server)
            port = parsed.port or 1527
            out["fastapi_port"] = port
            pid = None
            for cmd in (["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
                        ["ss", "-ltnp", "sport", "=", ":%d" % port]):
                try:
                    o = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                                 timeout=5).decode().strip()
                    if cmd[0] == "lsof":
                        if o:
                            pid = int(o.splitlines()[0])
                            out["fastapi_pid_via"] = "lsof"
                            break
                    else:
                        m = re.search(r"pid=(\d+)", o)
                        if m:
                            pid = int(m.group(1))
                            out["fastapi_pid_via"] = "ss"
                            break
                except Exception:
                    pass
            if pid:
                out["fastapi_pid"] = pid
                # /proc/<pid>/exe resolves to the *actual* interpreter
                # binary, which is normally the system Python — not the
                # venv. The venv's `bin/python` is a symlink to the same
                # system binary, so `exe` won't help us locate the venv.
                # Instead, parse /proc/<pid>/cmdline (null-separated
                # bytes) and use argv[0], which IS the venv path.
                try:
                    raw = open(f"/proc/{pid}/cmdline", "rb").read()
                    argv = [a.decode("utf-8", "replace")
                            for a in raw.split(b"\\x00") if a]
                    out["fastapi_cmdline"] = argv
                except Exception as e:
                    argv = []
                    out["fastapi_cmdline_err"] = repr(e)
                try:
                    exe_link = os.readlink(f"/proc/{pid}/exe")
                    out["fastapi_exe"] = exe_link
                except Exception as e:
                    out["fastapi_exe_err"] = repr(e)
                # Try to read the process's environ for VIRTUAL_ENV /
                # PATH — useful for paper provenance.
                try:
                    env_raw = open(f"/proc/{pid}/environ", "rb").read()
                    env_items = {}
                    for entry in env_raw.split(b"\\x00"):
                        if b"=" in entry:
                            k, _, v = entry.partition(b"=")
                            env_items[k.decode("utf-8", "replace")] = v.decode(
                                "utf-8", "replace"
                            )
                    out["fastapi_environ"] = {
                        k: env_items[k] for k in (
                            "VIRTUAL_ENV", "CONDA_PREFIX", "PATH",
                            "PYTHONPATH", "LD_LIBRARY_PATH",
                            "CUDA_VISIBLE_DEVICES", "NNI_MODEL_DIR",
                            "NNINTERACTIVE_DEVICE",
                        ) if k in env_items
                    }
                except Exception as e:
                    out["fastapi_environ_err"] = repr(e)
                # Pick the venv python: prefer argv[0] (e.g.
                # /media/volume/MyData/nninteractive/bin/python),
                # fall back to /proc/<pid>/exe.
                py_bin = None
                if argv and "python" in os.path.basename(argv[0]).lower():
                    py_bin = argv[0]
                elif "fastapi_exe" in out:
                    py_bin = out["fastapi_exe"]
                out["fastapi_py_bin"] = py_bin
                if py_bin:
                    def _run(cmd, timeout=30, scrub_dyn=False):
                        # subprocess.run never raises on non-zero exit,
                        # so we get full stdout+stderr regardless. We
                        # explicitly clear PYTHONPATH/PYTHONHOME because
                        # Slicer sets them to host-Python paths that
                        # break the venv. When scrub_dyn=True we ALSO
                        # clear LD_LIBRARY_PATH/DYLD_LIBRARY_PATH so the
                        # child doesn't inherit Slicer's bundled ITK/VTK
                        # (which conflicts with the venv's pip-installed
                        # ones and causes SIGBUS on import).
                        clean_env = dict(os.environ)
                        clean_env.pop("PYTHONPATH", None)
                        clean_env.pop("PYTHONHOME", None)
                        if scrub_dyn:
                            clean_env.pop("LD_LIBRARY_PATH", None)
                            clean_env.pop("DYLD_LIBRARY_PATH", None)
                            clean_env.pop("LD_PRELOAD", None)
                        proc = subprocess.run(
                            cmd, capture_output=True, timeout=timeout,
                            env=clean_env,
                        )
                        return {
                            "returncode": proc.returncode,
                            "stdout": proc.stdout.decode("utf-8", "replace"),
                            "stderr": proc.stderr.decode("utf-8", "replace"),
                        }
                    # Plain version probe (no heavy imports).
                    try:
                        ver = _run([py_bin, "-c",
                                    "import sys, platform; "
                                    "print(sys.version.split()[0]); "
                                    "print(sys.executable); "
                                    "print(platform.platform())"],
                                   timeout=15)
                        out["fastapi_python"] = ver
                    except Exception as e:
                        out["fastapi_python_err"] = repr(e)
                    # Heavy import probe (torch.cuda.is_available()
                    # etc). This will SIGBUS if the child inherits
                    # Slicer's LD_LIBRARY_PATH and the venv's torch
                    # links to a different ITK/VTK. We retry with
                    # scrub_dyn=True if the first attempt died.
                    probe_src = '''
import importlib, json
info = {}
for n in ('torch','nnInteractive','nnunetv2','nnunet',
          'SimpleITK','numpy','batchgenerators',
          'dynamic_network_architectures','fastapi','uvicorn'):
    try:
        m = importlib.import_module(n)
        info[n] = getattr(m,'__version__','?')
    except Exception as e:
        info[n] = f'<missing: {e!r}>'
try:
    import torch
    info['torch_cuda_available'] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info['torch_cuda_device_name'] = torch.cuda.get_device_name(0)
        info['torch_cuda_device_count'] = torch.cuda.device_count()
        info['torch_cuda_version'] = torch.version.cuda
    if hasattr(torch.backends, 'mps'):
        info['torch_mps_available'] = bool(torch.backends.mps.is_available())
except Exception as e:
    info['torch_err'] = repr(e)
print(json.dumps(info))
'''
                    fd, probe_path = tempfile.mkstemp(suffix=".py", prefix="bs_probe_")
                    os.close(fd)
                    try:
                        with open(probe_path, "w") as f:
                            f.write(probe_src)
                        # First attempt with the parent env (in case the
                        # libs are compatible).
                        probe_out = _run([py_bin, probe_path], timeout=120)
                        if probe_out["returncode"] != 0:
                            # Retry with cleared LD_LIBRARY_PATH so the
                            # venv's own libs win.
                            probe_out2 = _run([py_bin, probe_path],
                                              timeout=120, scrub_dyn=True)
                            out["fastapi_packages_raw_first"] = probe_out
                            out["fastapi_packages_raw"] = probe_out2
                            probe_out = probe_out2
                        else:
                            out["fastapi_packages_raw"] = probe_out
                        if probe_out["returncode"] == 0:
                            try:
                                out["fastapi_packages"] = json.loads(
                                    probe_out["stdout"].strip().splitlines()[-1]
                                )
                            except Exception as e:
                                out["fastapi_packages_parse_err"] = repr(e)
                    except Exception as e:
                        out["fastapi_packages_err"] = repr(e)
                    finally:
                        try:
                            os.unlink(probe_path)
                        except Exception:
                            pass
                    # pip freeze for full reproducibility.
                    try:
                        freeze = _run([py_bin, "-m", "pip", "freeze"], timeout=45)
                        if freeze["returncode"] == 0:
                            freeze_text = freeze["stdout"]
                            out["fastapi_pip_freeze"] = freeze_text
                            # Distil the key packages so paper readers
                            # have a quick-reference even without
                            # opening pip_freeze.
                            wanted = (
                                "torch", "nnInteractive", "nninteractive",
                                "nnunetv2", "nnunet", "SimpleITK",
                                "numpy", "batchgenerators",
                                "dynamic_network_architectures",
                                "fastapi", "uvicorn", "huggingface_hub",
                                "scipy", "scikit-image",
                            )
                            wanted_lc = {w.lower() for w in wanted}
                            distil = {}
                            for line in freeze_text.splitlines():
                                if "==" in line:
                                    name, _, ver = line.partition("==")
                                elif "@" in line:
                                    name, _, ver = line.partition("@")
                                else:
                                    continue
                                if name.strip().lower() in wanted_lc:
                                    distil[name.strip()] = ver.strip()
                            out["fastapi_packages_from_freeze"] = distil
                        else:
                            out["fastapi_pip_freeze_err"] = freeze
                    except Exception as e:
                        out["fastapi_pip_freeze_err"] = repr(e)
        except Exception as e:
            out["fastapi_introspect_err"] = repr(e)
    # nnInteractive Python package
    try:
        import nnInteractive
        out["nninteractive_version"] = getattr(nnInteractive, "__version__", "?")
        out["nninteractive_path"] = nnInteractive.__file__
    except Exception as e:
        out["nninteractive_err"] = repr(e)
    # PyTorch + device
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["torch_cuda_device_count"] = torch.cuda.device_count()
            out["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            out["torch_cuda_version"] = torch.version.cuda
        try:
            out["torch_mps_available"] = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
        except Exception:
            pass
    except Exception as e:
        out["torch_err"] = repr(e)
    # nnInteractive model directory + checkpoint hashes
    try:
        model_dir = os.environ.get("NNI_MODEL_DIR") or os.environ.get(
            "NNINTERACTIVE_MODEL_DIR"
        )
        if model_dir and os.path.isdir(model_dir):
            entries = []
            total = 0
            for root, dirs, files in os.walk(model_dir):
                for fn in sorted(files):
                    fp = os.path.join(root, fn)
                    try:
                        sz = os.path.getsize(fp)
                    except Exception:
                        sz = -1
                    rel = os.path.relpath(fp, model_dir)
                    if fn.endswith((".pth", ".pt", ".ckpt", ".json", ".yaml", ".yml")):
                        try:
                            h = hashlib.sha256()
                            with open(fp, "rb") as f:
                                while True:
                                    blk = f.read(1 << 20)
                                    if not blk: break
                                    h.update(blk)
                            entries.append({
                                "path": rel, "size": sz, "sha256": h.hexdigest()
                            })
                        except Exception as e:
                            entries.append({"path": rel, "size": sz,
                                            "sha256_err": repr(e)})
                    total += sz
            out["nninteractive_model_dir"] = model_dir
            out["nninteractive_model_total_bytes"] = total
            out["nninteractive_model_entries"] = entries
    except Exception as e:
        out["nninteractive_model_err"] = repr(e)
    # Process / system stats (best effort)
    try:
        import psutil
        proc = psutil.Process()
        out["proc_pid"] = proc.pid
        out["proc_rss_mb"] = round(proc.memory_info().rss / 1e6, 1)
        out["sys_total_mem_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        out["sys_available_mem_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
        out["sys_cpu_count"] = psutil.cpu_count(logical=True)
    except Exception as e:
        out["psutil_err"] = repr(e)
    out["status"] = "ok"
except Exception as e:
    out["status"] = "exception"
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
__execResult.update(out)
"""


# Hash the active volume's voxel content + record its essential identity.
# This is the single most important piece of provenance: without it,
# anyone re-running can't verify they're operating on the same image.
HASH_ACTIVE_VOLUME_SRC = """\
import slicer, hashlib, traceback
out = {}
try:
    sel = slicer.app.applicationLogic().GetSelectionNode()
    vol_id = sel.GetActiveVolumeID() if sel else None
    vol = slicer.mrmlScene.GetNodeByID(vol_id) if vol_id else None
    if vol is None:
        __execResult["status"] = "no_active_volume"
    else:
        arr = slicer.util.arrayFromVolume(vol)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        out["status"] = "ok"
        out["volume_id"] = vol.GetID()
        out["volume_name"] = vol.GetName()
        out["shape_kji"] = list(arr.shape)
        out["dtype"] = str(arr.dtype)
        out["nbytes"] = int(arr.nbytes)
        out["spacing_mm"] = [round(s, 6) for s in vol.GetSpacing()]
        out["origin"] = [round(o, 6) for o in vol.GetOrigin()]
        out["scalar_min"] = float(arr.min())
        out["scalar_max"] = float(arr.max())
        out["scalar_mean"] = float(arr.mean())
        out["sha256_voxels"] = h
        try:
            sn = vol.GetStorageNode()
            if sn and sn.GetFileName():
                out["source_filename"] = sn.GetFileName()
        except Exception:
            pass
        __execResult.update(out)
except Exception as e:
    __execResult["status"] = "exception"
    __execResult["error"] = repr(e)
    __execResult["traceback"] = traceback.format_exc()
"""


# Save the entire active segmentation as one or more NIfTI files and
# stream them back base64-encoded. We export each segment as its own
# binary labelmap (.nii.gz) AND a single multi-label composite — the
# composite is what most downstream tools expect, while per-segment
# files preserve the structure-by-structure identity.
EXPORT_SEGMENTATION_SRC = """\
import slicer, os, tempfile, base64, hashlib, traceback, vtk
out = {"per_segment": [], "composite": None}
def _hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()
def _read_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")
try:
    seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    seg_node = None
    for n in seg_nodes:
        if "do not touch" in n.GetName().lower():
            continue
        seg_node = n
        break
    if seg_node is None:
        __execResult["status"] = "no_segmentation"
    else:
        sel = slicer.app.applicationLogic().GetSelectionNode()
        ref_vol = slicer.mrmlScene.GetNodeByID(sel.GetActiveVolumeID()) if sel else None
        td = tempfile.mkdtemp(prefix="bs_export_")
        # Per-segment binary labelmaps
        seg = seg_node.GetSegmentation()
        for ii in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(ii)
            s = seg.GetSegment(sid)
            sname = s.GetName()
            # Create a temp labelmap volume node containing only this segment
            label_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", f"_export_{sid}"
            )
            try:
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(sid)
                ok = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    seg_node, ids, label_node, ref_vol
                )
                if not ok:
                    out["per_segment"].append({"sid": sid, "name": sname,
                                                "error": "ExportSegmentsToLabelmapNode returned False"})
                else:
                    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in sid)
                    fp = os.path.join(td, f"{safe}.nii.gz")
                    saved = slicer.util.saveNode(label_node, fp)
                    if saved and os.path.exists(fp):
                        size = os.path.getsize(fp)
                        out["per_segment"].append({
                            "sid": sid, "name": sname,
                            "filename": f"{safe}.nii.gz",
                            "size_bytes": size,
                            "sha256": _hash_file(fp),
                            "data_b64": _read_b64(fp),
                            "color": [round(c, 4) for c in s.GetColor()],
                        })
                    else:
                        out["per_segment"].append({"sid": sid, "name": sname,
                                                    "error": "saveNode failed"})
            finally:
                slicer.mrmlScene.RemoveNode(label_node)
        # Composite multi-label.
        # Slicer 5.x signature is
        #   ExportAllSegmentsToLabelmapNode(segNode, labelmapNode,
        #                                   extentComputationMode=2)
        # — there's NO reference-volume argument here (it picks up the
        # segmentation's source volume automatically).
        composite_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_export_composite"
        )
        try:
            ok = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                seg_node, composite_node
            )
            if ok:
                fp = os.path.join(td, "composite.nii.gz")
                saved = slicer.util.saveNode(composite_node, fp)
                if saved and os.path.exists(fp):
                    out["composite"] = {
                        "filename": "composite.nii.gz",
                        "size_bytes": os.path.getsize(fp),
                        "sha256": _hash_file(fp),
                        "data_b64": _read_b64(fp),
                    }
            else:
                out["composite_err"] = "ExportAllSegmentsToLabelmapNode returned False"
        except Exception as e:
            out["composite_err"] = repr(e)
        finally:
            slicer.mrmlScene.RemoveNode(composite_node)
        # Cleanup
        try:
            for fn in os.listdir(td):
                os.unlink(os.path.join(td, fn))
            os.rmdir(td)
        except Exception:
            pass
        out["status"] = "ok"
        out["segmentation_node_name"] = seg_node.GetName()
        out["segment_count"] = seg.GetNumberOfSegments()
        __execResult.update(out)
except Exception as e:
    __execResult["status"] = "exception"
    __execResult["error"] = repr(e)
    __execResult["traceback"] = traceback.format_exc()
"""


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

def _now_id(label: Optional[str] = None) -> str:
    stamp = datetime.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    if label:
        safe = "".join(c if c.isalnum() or c in "_-" else "-" for c in label)
        return f"{stamp}_{safe}_{short}"
    return f"{stamp}_{short}"


class RunLogger:
    """Append-only structured logging into a single run directory.

    Usage:
        logger = RunLogger.start(out_dir, args=vars(parsed_args))
        logger.log("...")
        logger.event("step_pick", step=0, ijk=[1,2,3])
        logger.write_artifact("artifacts/composite.nii.gz", data_bytes)
        logger.finalize(stop_reason={"reason": "saturated", ...})
    """

    def __init__(self, root: Path, run_id: str, args: dict, t0: float):
        self.root = Path(root)
        self.run_id = run_id
        self.args = args
        self.t0 = t0
        self.events_path = self.root / "events.jsonl"
        self.log_path = self.root / "log.txt"
        self.manifest_path = self.root / "manifest.json"
        self.environment_path = self.root / "environment.json"
        self.inputs_path = self.root / "inputs.json"
        self.artifacts_dir = self.root / "artifacts"
        self.artifact_index_path = self.artifacts_dir / "index.json"
        self.stop_reason_path = self.root / "stop_reason.json"
        self.replay_path = self.root / "replay.sh"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "a", buffering=1, encoding="utf-8")
        self._artifact_index: list[dict] = []

    @classmethod
    def start(cls, root: Path, args: dict, label: Optional[str] = None,
              run_id: Optional[str] = None) -> "RunLogger":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        run_id = run_id or _now_id(label)
        t0 = time.time()
        self = cls(root=root, run_id=run_id, args=args, t0=t0)
        manifest = {
            "run_id": run_id,
            "label": label,
            "started_utc": datetime.datetime.now(UTC).isoformat(),
            "started_unix": t0,
            "args": args,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        self.event("run_start", **manifest)
        return self

    # --- event log ---------------------------------------------------

    def event(self, kind: str, **payload: Any) -> dict:
        rec = {
            "t": datetime.datetime.now(UTC).isoformat(),
            "ts_unix": round(time.time(), 4),
            "elapsed_s": round(time.time() - self.t0, 4),
            "event": kind,
            **payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def log(self, msg: str = "", echo: bool = True) -> None:
        ts = datetime.datetime.now(UTC).isoformat()
        line = f"[{ts}] {msg}" if msg else ""
        self._log_fh.write(line + "\n")
        if echo:
            print(msg)

    # --- environment + inputs ----------------------------------------

    def record_local_env(self, repo_root: Optional[Path] = None) -> dict:
        env = capture_local_env(repo_root=repo_root)
        self.environment_path.write_text(
            json.dumps({"local": env}, indent=2, default=str)
        )
        self.event("local_env", **env)
        return env

    def record_remote_env(self, remote_env: dict) -> None:
        existing = {}
        if self.environment_path.exists():
            try:
                existing = json.loads(self.environment_path.read_text())
            except Exception:
                existing = {}
        existing["remote"] = remote_env
        self.environment_path.write_text(
            json.dumps(existing, indent=2, default=str)
        )
        self.event("remote_env", **remote_env)

    def record_inputs(self, inputs: dict) -> None:
        self.inputs_path.write_text(json.dumps(inputs, indent=2, default=str))
        self.event("inputs", **inputs)

    # --- artifacts ---------------------------------------------------

    def write_artifact(self, relpath: str, data: bytes,
                       kind: Optional[str] = None,
                       extra: Optional[dict] = None) -> dict:
        rel = Path(relpath)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rec = {
            "path": str(rel.as_posix()),
            "kind": kind,
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        if extra:
            rec["extra"] = extra
        self._artifact_index.append(rec)
        self.artifact_index_path.write_text(
            json.dumps(self._artifact_index, indent=2, default=str)
        )
        # Avoid colliding with the event's own "kind" parameter — record
        # the artifact kind under "artifact_kind" inside the event log.
        evt_payload = {k: v for k, v in rec.items() if k != "kind"}
        evt_payload["artifact_kind"] = rec.get("kind")
        self.event("artifact", **evt_payload)
        return rec

    # --- replay ------------------------------------------------------

    def write_replay(self, command: list[str], env_keys: Iterable[str] = ()) -> None:
        env_lines = []
        for k in env_keys:
            v = os.environ.get(k)
            if v is None:
                continue
            if any(p in k for p in ("TOKEN", "KEY", "SECRET", "PASS")):
                v = "<<set me>>"
            env_lines.append(f'export {k}="{v}"')
        cmd = " \\\n    ".join(command)
        env_block = "\n".join(env_lines) if env_lines else "# no env vars to export"
        body = (
            "#!/usr/bin/env bash\n"
            f"# Reproduction script for run {self.run_id}.\n"
            "# Generated by run_telemetry.RunLogger.write_replay().\n"
            "#\n"
            "# Verify that:\n"
            "#   1. Your remote Slicer instance has a volume whose\n"
            "#      sha256_voxels matches inputs.json's sha256_voxels.\n"
            "#   2. SLICER_WEBSERVER_URL points at a Slicer instance with\n"
            "#      the SlicerNNInteractive plugin running.\n"
            "set -euo pipefail\n"
            'cd "$(dirname "$0")/../.."  # back to repo root\n'
            "\n"
            f"{env_block}\n"
            "\n"
            f"{cmd}\n"
        )
        self.replay_path.write_text(body)
        try:
            self.replay_path.chmod(0o755)
        except Exception:
            pass

    # --- finalize ----------------------------------------------------

    def finalize(self, stop_reason: Optional[dict] = None,
                 summary: Optional[dict] = None) -> None:
        if stop_reason is not None:
            self.stop_reason_path.write_text(
                json.dumps(stop_reason, indent=2, default=str)
            )
            self.event("stop_reason", **stop_reason)
        if summary is not None:
            (self.root / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
        self.event("run_end", duration_s=round(time.time() - self.t0, 3))
        try:
            self._log_fh.close()
        except Exception:
            pass
