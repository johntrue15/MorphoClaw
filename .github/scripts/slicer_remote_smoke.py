#!/usr/bin/env python3
"""
Smoke test for driving 3D Slicer's built-in Web Server module from a
remote machine.

Runs three escalating checks against ``$NNI_REMOTE_URL`` (or
``$SLICER_WEBSERVER_URL`` if set) and writes a small bundle of artifacts
to ``smoke_out/`` so we can visually confirm "screen data" came back:

  1) ``GET /slicer/`` returns HTTP 200 + the static index.html.
       (Confirms reachability through the Exosphere proxy.)
  2) ``POST /slicer/exec`` runs trivial Python in Slicer's process:
       sets ``__execResult["greeting"] = "hello from <slicer version>"``
       and ``__execResult["sum"] = 2 + 2``.
       (Confirms the exec channel + JSON round-trip.)
  3) ``POST /slicer/exec`` captures the current Slicer main-window
     screenshot via ``slicer.util.mainWindow().grab()``, returns a
     base64-encoded PNG, and we save it to ``smoke_out/slicer_screen.png``.
       (Confirms "screen data is flowing into the browser/client".)

If you want to skip step 3 (e.g. because Slicer is in headless mode),
pass ``--no-screenshot``.

Usage
-----
    export NNI_REMOTE_URL='https://http-149-165-155-167-2016.proxy-js2-iu.exosphere.app/'
    python3 .github/scripts/slicer_remote_smoke.py

The URL must point at the *Slicer Web Server* (default port 2016), not
the SlicerNNInteractive FastAPI server (port 1527). They are different
services. ``NNI_REMOTE_URL`` is reused for convenience; if you have both
set up at once, prefer ``SLICER_WEBSERVER_URL``.

Setup checklist (one-time, inside the Guacamole Slicer GUI)
-----------------------------------------------------------
1. Module dropdown -> Servers -> Web Server
2. Expand "Advanced" section
3. Tick "Slicer API exec" (the security warning is expected)
4. Click "Start server"
5. The log should say:  "Starting server on port 2016"
6. Confirm a green "Server running" indicator
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OUT_DIR = Path("smoke_out")


# ---------------------------------------------------------------------------
# Tiny HTTP helpers (stdlib only — no requests dependency)
# ---------------------------------------------------------------------------

def _read_url() -> str:
    url = (
        os.environ.get("SLICER_WEBSERVER_URL", "").strip()
        or os.environ.get("NNI_REMOTE_URL", "").strip()
        or os.environ.get("NNI_REMOTE_WS", "").strip()
    )
    if not url:
        sys.exit(
            "ERROR: SLICER_WEBSERVER_URL (or NNI_REMOTE_URL) is not set.\n"
            "  example:\n"
            "    export SLICER_WEBSERVER_URL='https://http-149-165-155-167-2016"
            ".proxy-js2-iu.exosphere.app/'"
        )
    if url.startswith("ws://"):
        url = "http://" + url[len("ws://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    return url.rstrip("/")


def http_get(url: str, timeout: float = 10) -> tuple[int, dict, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def http_post_python(url: str, source: str, timeout: float = 60
                     ) -> tuple[int, dict, bytes]:
    """POST raw Python source to /slicer/exec and return (status, headers, body)."""
    body = source.encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "text/plain"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

def check_reachable(base_url: str) -> bool:
    """Confirm the proxy can reach Slicer's Web Server.

    Slicer's static handler serves the docroot at ``/`` (returns 200), and
    routes ``/slicer/<command>`` to the SlicerRequestHandler. The bare
    prefix ``/slicer/`` without a command currently returns HTTP 500
    (handler invoked with empty command). We probe both ``/`` (the
    docroot) and a known good API endpoint (``/slicer/screenshot``) to
    confirm the server is listening AND the SlicerRequestHandler is
    enabled.
    """
    probes = [
        ("/", "static docroot"),
        ("/slicer/screenshot", "Slicer API: screenshot endpoint"),
    ]
    all_ok = True
    for path, label in probes:
        status, _, body = http_get(base_url + path, timeout=8)
        ok = status == 200 and len(body) > 0
        flag = "OK " if ok else "FAIL"
        print(f"      [{flag}] GET {path:25s}  HTTP {status}  body={len(body)} bytes  ({label})")
        if not ok:
            all_ok = False
    return all_ok


def check_exec(base_url: str) -> bool:
    print(f"[2/3] POST {base_url}/slicer/exec  (trivial Python)…")
    src = textwrap.dedent("""
        import slicer
        __execResult["greeting"] = (
            f"hello from Slicer {slicer.app.applicationVersion}"
        )
        __execResult["python_version"] = (
            __import__('sys').version.split()[0]
        )
        __execResult["sum"] = 2 + 2
        try:
            __execResult["loaded_modules"] = sorted(list(slicer.modules.names))[:25]
        except Exception:
            __execResult["loaded_modules"] = []
    """).strip()
    t0 = time.time()
    status, _, body = http_post_python(base_url + "/slicer/exec", src, timeout=30)
    print(f"      HTTP {status}  ({(time.time()-t0)*1e3:.0f} ms)  body={len(body)} bytes")
    if status != 200:
        print(f"      body preview: {body[:300]!r}")
        return False
    try:
        result = json.loads(body)
    except Exception as e:
        print(f"      ERROR: response not valid JSON ({e})")
        print(f"      body preview: {body[:300]!r}")
        return False
    print(f"      Slicer says: {result.get('greeting')!r}")
    print(f"      python:       {result.get('python_version')!r}")
    print(f"      2 + 2 =       {result.get('sum')!r}")
    mods = result.get("loaded_modules") or []
    if mods:
        print(f"      first 25 modules: {', '.join(mods)}")
    return result.get("sum") == 4


def check_screenshot(base_url: str, out_dir: Path) -> bool:
    print(f"[3/3] POST {base_url}/slicer/exec  (capture screenshot)…")
    src = textwrap.dedent("""
        import slicer, qt, base64
        from PythonQt.QtCore import QBuffer, QByteArray, QIODevice

        mw = slicer.util.mainWindow()
        if mw is None:
            __execResult["error"] = "no main window (running headless?)"
        else:
            pix = mw.grab()
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.WriteOnly)
            pix.save(buf, b"PNG")
            buf.close()
            png_bytes = bytes(ba.data())
            __execResult["png_b64"] = base64.b64encode(png_bytes).decode("ascii")
            __execResult["png_size"] = len(png_bytes)
            __execResult["window_size"] = [int(mw.width), int(mw.height)]
    """).strip()
    t0 = time.time()
    status, _, body = http_post_python(base_url + "/slicer/exec", src, timeout=60)
    dt_ms = (time.time() - t0) * 1e3
    print(f"      HTTP {status}  ({dt_ms:.0f} ms)  body={len(body)} bytes")
    if status != 200:
        print(f"      body preview: {body[:300]!r}")
        return False
    try:
        result = json.loads(body)
    except Exception as e:
        print(f"      ERROR: response not valid JSON ({e})")
        return False
    if result.get("error"):
        print(f"      Slicer reported: {result['error']}")
        return False
    b64 = result.get("png_b64", "")
    if not b64:
        print("      no png_b64 in response")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slicer_screen.png"
    out_path.write_bytes(base64.b64decode(b64))
    w, h = result.get("window_size", [0, 0])
    print(f"      window: {w}x{h}, PNG size: {result.get('png_size')} bytes")
    print(f"      saved screenshot -> {out_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="where to drop the screenshot (default: smoke_out/)")
    p.add_argument("--no-screenshot", action="store_true",
                   help="skip step 3 (useful if running headless)")
    args = p.parse_args(argv)

    base_url = _read_url()
    print(f"=== Slicer Web Server smoke test ===")
    print(f"target: {base_url}")
    print()
    print(f"[1/3] reachability probes:")
    ok = check_reachable(base_url)
    if not ok:
        print()
        print("FAILED at step 1/3: server not reachable.")
        print("  - Is Slicer's 'Web Server' module started? (Module dropdown -> Servers -> Web Server -> Start server)")
        print("  - Did the proxy URL match the port Slicer printed (default 2016)?")
        print("  - Try opening the URL in your browser to compare.")
        return 1

    ok = check_exec(base_url)
    if not ok:
        print()
        print("FAILED at step 2/3: /slicer/exec did not run our Python.")
        print("  - Did you tick 'Slicer API exec' in the Web Server module's Advanced section?")
        print("    (off by default; required for /exec)")
        print("  - Try the URL in a browser; the static index.html should still load.")
        return 2

    if args.no_screenshot:
        print()
        print("OK (skipped screenshot per --no-screenshot)")
        return 0

    ok = check_screenshot(base_url, args.out_dir)
    if not ok:
        print()
        print("WARNING at step 3/3: screenshot capture failed, but the exec channel works.")
        print("  - Slicer might be running without a Qt main window (headless).")
        print("  - We can still drive nnInteractive without screenshots.")
        return 3

    print()
    print("ALL THREE SMOKE CHECKS PASSED.")
    print(f"  Screenshot: {args.out_dir / 'slicer_screen.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
