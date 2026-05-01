"""
Install MONAIAuto3DSeg extension into 3D Slicer.

Run this ONCE inside 3D Slicer to bootstrap the MONAI extension:

    /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --python-script slicer_install_monai.py

After installation, restart Slicer for the extension to load.
On first segmentation run, PyTorch + MONAI packages are auto-installed
(~5-10 min, needs ~20GB free disk space).
"""
import slicer
import sys
import os

def install_extension():
    em = slicer.app.extensionsManagerModel()
    ext_name = "MONAIAuto3DSeg"

    if em.isExtensionInstalled(ext_name):
        print(f"[OK] {ext_name} is already installed.")
        print(f"     Extension path: {em.extensionInstallPath(ext_name)}")
        slicer.app.exit(0)
        return

    print(f"Installing {ext_name} extension...")
    print("This requires an internet connection and may take a few minutes.")

    metadata = em.retrieveExtensionMetadataByName(ext_name)
    if not metadata or not metadata.get("extension_id"):
        print(f"[ERROR] Could not find {ext_name} in the extensions catalog.")
        print("        Make sure you have internet access and are using a compatible Slicer version.")
        slicer.app.exit(1)
        return

    ext_id = metadata["extension_id"]
    print(f"Found extension ID: {ext_id}")

    success = em.downloadAndInstallExtensionByName(ext_name)
    if success:
        print(f"[OK] {ext_name} installed successfully.")
        print("     RESTART 3D Slicer for the extension to load.")
    else:
        print(f"[ERROR] Failed to install {ext_name}.")
        print("        Check Slicer logs for details.")
        slicer.app.exit(1)
        return

    # Also install PyTorch Util (dependency for GPU/CPU selection)
    pytorch_ext = "PyTorch"
    if not em.isExtensionInstalled(pytorch_ext):
        print(f"Installing {pytorch_ext} extension (dependency)...")
        em.downloadAndInstallExtensionByName(pytorch_ext)

    slicer.app.exit(0)


install_extension()
