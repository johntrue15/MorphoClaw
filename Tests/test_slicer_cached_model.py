"""SlicerMorph cached-model regression tests.

Three tiers, smallest first:

1. **Fixture sanity** (always runs).  Parses the tiny ASCII tetrahedron
   shipped in ``Tests/fixtures/tetrahedron.ply`` and asserts the
   header, vertex count, and face count are intact.  This catches
   accidental corruption of the deterministic test asset.

2. **Cache discovery** (runs if the real MorphoSource cache is on
   disk).  Walks ``data/morphosource-download-*/`` looking for the
   chameleon skull PLY that the project pinned as a known-good model,
   then asserts it begins with a valid PLY header.  Skipped (not
   failed) on cloud runners where the 112 MB binary isn't checked in.

3. **Headless Slicer load** (runs only if ``SLICER_BIN`` exists on the
   host).  Spawns ``Slicer --no-splash --no-main-window
   --python-script slicer_load_test.py`` against the cached PLY and
   parses the JSON result.  Verifies that ``slicer.util.loadModel``
   succeeded and the mesh has non-degenerate bounds.  When the
   SlicerMorph extension is also importable, asserts that too.

Run the lightweight tiers with::

    pytest Tests/test_slicer_cached_model.py

To force the full integration tier::

    SLICER_BIN=/Applications/Slicer.app/Contents/MacOS/Slicer \\
    SLICER_REQUIRE_INTEGRATION=1 \\
    pytest Tests/test_slicer_cached_model.py -k slicer_loads
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PLY = REPO_ROOT / "Tests" / "fixtures" / "tetrahedron.ply"
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
LOAD_TEST_SCRIPT = SCRIPTS_DIR / "slicer_load_test.py"
DATA_DIR = REPO_ROOT / "data"

# Default cache locations: the project ships one large chameleon-skull PLY
# under ``data/morphosource-download-000769445/``.  We also accept any
# similarly-named cached download so future fixtures don't need test edits.
CACHED_PLY_GLOB = "morphosource-download-*/**/*.ply"


def _slicer_bin() -> Path | None:
    candidate = os.environ.get("SLICER_BIN", "/Applications/Slicer.app/Contents/MacOS/Slicer")
    path = Path(candidate)
    return path if path.exists() else None


def _find_cached_morphosource_ply() -> Path | None:
    """Return the most plausible cached MorphoSource PLY, or None."""
    explicit = os.environ.get("SLICER_TEST_MODEL")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    if not DATA_DIR.exists():
        return None
    matches = sorted(DATA_DIR.glob(CACHED_PLY_GLOB))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Tier 1: deterministic fixture (always runs in CI)
# ---------------------------------------------------------------------------


def test_tetrahedron_fixture_exists() -> None:
    assert FIXTURE_PLY.exists(), (
        f"Test fixture missing: {FIXTURE_PLY}.  This file is required by "
        "the SlicerMorph load test and must remain in version control."
    )


def test_tetrahedron_fixture_is_valid_ascii_ply() -> None:
    text = FIXTURE_PLY.read_text(encoding="utf-8").splitlines()
    assert text[0] == "ply", "PLY magic missing"
    assert text[1].startswith("format ascii"), "fixture must remain ASCII for portability"

    end_header_idx = next(
        (i for i, line in enumerate(text) if line.strip() == "end_header"),
        -1,
    )
    assert end_header_idx > 0, "end_header line missing"

    header = text[: end_header_idx + 1]
    body = [line for line in text[end_header_idx + 1 :] if line.strip()]

    vert_match = next(
        (
            re.match(r"element vertex (\d+)", line)
            for line in header
            if line.startswith("element vertex")
        ),
        None,
    )
    face_match = next(
        (
            re.match(r"element face (\d+)", line)
            for line in header
            if line.startswith("element face")
        ),
        None,
    )
    assert vert_match and face_match, "fixture must declare vertex + face elements"

    n_verts = int(vert_match.group(1))
    n_faces = int(face_match.group(1))
    assert (
        len(body) == n_verts + n_faces
    ), f"body line count ({len(body)}) != declared verts+faces ({n_verts}+{n_faces})"

    # Spot-check vertex parses as 3 floats and faces parse as `count + indices`.
    for vline in body[:n_verts]:
        parts = vline.split()
        assert len(parts) == 3, f"bad vertex line: {vline!r}"
        [float(p) for p in parts]

    for fline in body[n_verts : n_verts + n_faces]:
        parts = fline.split()
        assert len(parts) >= 4, f"bad face line: {fline!r}"
        count = int(parts[0])
        assert (
            len(parts) == count + 1
        ), f"face declares {count} verts but has {len(parts) - 1} indices"


# ---------------------------------------------------------------------------
# Tier 2: real MorphoSource cache discovery (runs if the cache is on disk)
# ---------------------------------------------------------------------------


def test_morphosource_cache_discovery_returns_ply_or_skips() -> None:
    cached = _find_cached_morphosource_ply()
    if not cached:
        pytest.skip(
            "No MorphoSource cached PLY found under data/morphosource-download-*. "
            "This is expected on cloud CI runners; the file is in .gitignore."
        )
    assert cached.is_file(), f"discovered path is not a file: {cached}"
    assert (
        cached.stat().st_size > 1024
    ), f"cached model suspiciously small ({cached.stat().st_size} bytes): {cached}"

    with cached.open("rb") as fh:
        magic = fh.readline().strip()
    assert magic == b"ply", f"cached model does not start with PLY magic: {cached} (got {magic!r})"


def test_cached_ply_header_advertises_geometry() -> None:
    cached = _find_cached_morphosource_ply()
    if not cached:
        pytest.skip("No cached MorphoSource PLY on this runner.")

    header = b""
    with cached.open("rb") as fh:
        while b"end_header" not in header:
            chunk = fh.readline()
            if not chunk:
                break
            header += chunk
            if len(header) > 8192:
                break

    text = header.decode("ascii", errors="replace")
    assert "element vertex" in text, f"PLY header missing vertex element: {cached}"
    assert "element face" in text, f"PLY header missing face element: {cached}"
    vert_match = re.search(r"element vertex (\d+)", text)
    face_match = re.search(r"element face (\d+)", text)
    assert vert_match and int(vert_match.group(1)) > 0
    assert face_match and int(face_match.group(1)) > 0


# ---------------------------------------------------------------------------
# Tier 3: headless 3D Slicer integration (runs only when Slicer is installed)
# ---------------------------------------------------------------------------


_slicer_bin_path = _slicer_bin()
slicer_required = os.environ.get("SLICER_REQUIRE_INTEGRATION") == "1"
SLICER_SKIP = pytest.mark.skipif(
    _slicer_bin_path is None and not slicer_required,
    reason="3D Slicer not installed on this runner (set SLICER_BIN to enable).",
)


@SLICER_SKIP
def test_slicer_load_test_script_exists() -> None:
    assert LOAD_TEST_SCRIPT.exists(), (
        f"Headless loader script missing: {LOAD_TEST_SCRIPT}. "
        "This file is invoked by the SlicerMorph integration tests."
    )


SLICER_TIMEOUT_S = int(os.environ.get("SLICER_TEST_TIMEOUT_S", "180"))


def _run_slicer_load(model_path: Path, tmp_path: Path, require_slicermorph: bool = False) -> dict:
    """Spawn Slicer headlessly to load ``model_path`` and return the JSON result.

    Uses ``--testing --no-splash --no-main-window`` which is the same
    combination ``slicer_advanced_analysis.py`` relies on in production.
    Falls the test (rather than the whole CI job) if Slicer hangs past
    ``SLICER_TEST_TIMEOUT_S``.
    """
    slicer_bin = _slicer_bin()
    if slicer_bin is None:
        pytest.skip("SLICER_BIN does not point at an existing executable")

    out_path = tmp_path / "load_result.json"
    env = os.environ.copy()
    env["SLICER_LOAD_PATH"] = str(model_path)
    env["SLICER_LOAD_OUT"] = str(out_path)
    env["SLICER_MIN_POINTS"] = "1"
    if require_slicermorph:
        env["SLICER_REQUIRE_SLICERMORPH"] = "1"
    # Force a software-GL backend on headless runners where there is no display.
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    cmd = [
        str(slicer_bin),
        "--testing",
        "--no-splash",
        "--no-main-window",
        "--python-script",
        str(LOAD_TEST_SCRIPT),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SLICER_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"Slicer load probe timed out after {SLICER_TIMEOUT_S}s.\n"
            f"Model: {model_path}\n"
            f"Partial stdout:\n{(exc.stdout or b'')[-500:]!r}\n"
            f"Partial stderr:\n{(exc.stderr or b'')[-500:]!r}"
        )

    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    # Fallback: try to recover a JSON blob from stdout.
    stdout = proc.stdout or ""
    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(stdout[start : end + 1])
        except json.JSONDecodeError:
            pass

    pytest.fail(
        "Slicer load probe produced no parseable JSON.\n"
        f"exit_code={proc.returncode}\n"
        f"stdout tail:\n{stdout[-500:]}\n"
        f"stderr tail:\n{(proc.stderr or '')[-500:]}"
    )


@SLICER_SKIP
def test_slicer_loads_tetrahedron_fixture(tmp_path: Path) -> None:
    """Slicer must be able to load the tiny shipped fixture."""
    result = _run_slicer_load(FIXTURE_PLY, tmp_path)
    assert result.get("ok") is True, f"Slicer failed to load fixture: {result}"
    assert result.get("n_points") == 4, f"expected 4 vertices, got {result.get('n_points')}"
    assert result.get("n_cells") == 4, f"expected 4 faces, got {result.get('n_cells')}"


@SLICER_SKIP
def test_slicer_loads_cached_morphosource_model(tmp_path: Path) -> None:
    """Slicer must be able to load the real cached chameleon skull."""
    cached = _find_cached_morphosource_ply()
    if cached is None:
        pytest.skip("No cached MorphoSource PLY on this runner.")
    result = _run_slicer_load(cached, tmp_path)
    assert result.get("ok") is True, f"Slicer failed to load {cached}: {result}"
    assert (
        result.get("n_points", 0) > 1000
    ), f"cached model loaded with too few points ({result.get('n_points')}): {cached}"
    extent = result.get("extent") or [0, 0, 0]
    assert any(e > 0 for e in extent), f"loaded mesh has degenerate bounds: extent={extent}"


@SLICER_SKIP
def test_slicermorph_extension_available(tmp_path: Path) -> None:
    """When Slicer is installed we expect SlicerMorph to be installed too."""
    cached = _find_cached_morphosource_ply() or FIXTURE_PLY
    result = _run_slicer_load(cached, tmp_path)
    if not result.get("slicermorph_available"):
        if slicer_required:
            pytest.fail(
                "SlicerMorph extension not available: " f"{result.get('slicermorph_error')!r}"
            )
        pytest.skip(
            "SlicerMorph extension not available on this Slicer install. "
            "Install via Extensions Manager -> SlicerMorph."
        )
    assert result.get("slicermorph_available") is True


# ---------------------------------------------------------------------------
# Diagnostic helper -- useful when triaging skips on a new runner
# ---------------------------------------------------------------------------


def test_environment_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Emit a one-shot environment summary for the GitHub step log."""
    cached = _find_cached_morphosource_ply()
    print("SlicerMorph test environment:")
    print(f"  SLICER_BIN        = {os.environ.get('SLICER_BIN', '(unset)')}")
    print(f"  slicer_present    = {bool(_slicer_bin())}")
    print(f"  fixture_present   = {FIXTURE_PLY.exists()}")
    print(f"  cached_model      = {cached}")
    print(f"  loader_script     = {LOAD_TEST_SCRIPT}")
    captured = capsys.readouterr()
    # Always pass; this just emits diagnostics into the test log.
    assert "SlicerMorph test environment" in captured.out
