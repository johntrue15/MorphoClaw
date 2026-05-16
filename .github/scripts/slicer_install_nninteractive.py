"""
Install the SlicerNNInteractive extension into 3D Slicer.

This script enables the *visual* nnInteractive workflow inside 3D Slicer.
The headless / autoresearch-claw "paint" loop (.github/scripts/nninteractive_loop.py)
does NOT require this extension — it drives the nnInteractive Python API
directly. Install this only if a researcher wants to inspect or refine
the resulting segmentations interactively in Slicer.

Run ONCE inside 3D Slicer:

    /Applications/Slicer.app/Contents/MacOS/Slicer \
        --no-splash --python-script slicer_install_nninteractive.py

After installation, restart Slicer for the extension to load. The first
time the extension talks to a server it will prompt you for the URL of a
running `nninteractive-slicer-server` (default: http://localhost:1527).
The server is bootstrapped separately by `install_nninteractive.sh`.
"""
import os
import sys

import slicer


# Catalog name of the community-maintained 3D Slicer extension.
EXT_NAME = "SlicerNNInteractive"

# Optional helper extension that is also nice to have alongside (PyTorch util
# is used by other ML extensions; harmless if it's already installed).
OPTIONAL_EXTS = ("PyTorch",)


def install_extension(name: str) -> bool:
    em = slicer.app.extensionsManagerModel()
    if em.isExtensionInstalled(name):
        path = em.extensionInstallPath(name)
        print(f"[OK] {name} already installed at: {path}")
        return True

    print(f"Installing {name} extension (requires internet)...")
    metadata = em.retrieveExtensionMetadataByName(name)
    if not metadata or not metadata.get("extension_id"):
        print(f"[ERROR] {name} not found in the extensions catalog.")
        print("        Check internet access and that this Slicer build is")
        print("        recent enough (SlicerNNInteractive requires Slicer 5.6+).")
        return False

    print(f"  extension_id = {metadata['extension_id']}")
    if em.downloadAndInstallExtensionByName(name):
        print(f"[OK] {name} installed.")
        return True
    print(f"[ERROR] Failed to install {name}. Check Slicer logs.")
    return False


def main() -> int:
    ok = install_extension(EXT_NAME)
    for opt in OPTIONAL_EXTS:
        try:
            install_extension(opt)
        except Exception as exc:
            print(f"[WARN] Optional {opt}: {exc}")

    if ok:
        print("")
        print("Done. RESTART 3D Slicer to load SlicerNNInteractive.")
        print("Then open the 'nnInteractive' module and set the server URL")
        print(f"to whatever NNINTERACTIVE_SERVER_URL is on this host")
        print("(default: http://localhost:1527).")
        slicer.app.exit(0)
        return 0

    slicer.app.exit(1)
    return 1


sys.exit(main())
