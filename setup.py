"""
InstantMesh — Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  — path to Modly's embedded Python
    ext_dir     — absolute path to this extension directory
    gpu_sm      — GPU compute capability as integer (e.g. 89 for RTX 4050)
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(python_exe: str, ext_dir: Path, gpu_sm: int) -> None:
    venv   = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0"]
        print(f"[setup] SM {gpu_sm} (Blackwell) → PyTorch 2.7 + CUDA 12.8")
    elif gpu_sm >= 70:
        # Ada / Ampere / Turing — RTX 20/30/40 series, includes RTX 4050 (SM 89)
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print(f"[setup] SM {gpu_sm} → PyTorch 2.6 + CUDA 12.4")
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print(f"[setup] SM {gpu_sm} (legacy) → PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch…")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers — prebuilt wheel, no compilation needed
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers…")
    if gpu_sm >= 70:
        # PyTorch 2.6 + CUDA 12.4 prebuilt xformers
        pip(venv, "install", "xformers==0.0.29.post1", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Windows Triton (xformers dependency, prebuilt .whl — no MSVC needed)
    # ------------------------------------------------------------------ #
    if is_win:
        print("[setup] Installing Windows prebuilt triton…")
        triton_whl = (
            "https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/"
            "triton-2.0.0-cp310-cp310-win_amd64.whl"
        )
        try:
            pip(venv, "install", triton_whl)
        except subprocess.CalledProcessError:
            print("[setup] Triton whl install failed — skipping (xformers may still work).")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies…")
    pip(venv, "install",
        "diffusers==0.27.2",
        "transformers>=4.40.0",
        "accelerate",
        "huggingface_hub",
        "omegaconf",
        "einops",
        "Pillow",
        "numpy",
        "scipy",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "opencv-python-headless",
        "tqdm",
        "safetensors",
    )

    # ------------------------------------------------------------------ #
    # rembg
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg…")
    if gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
    else:
        pip(venv, "install", "rembg", "onnxruntime")

    # ------------------------------------------------------------------ #
    # onnxruntime (rembg dependency, sometimes missing)
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            pip(venv, "install", "onnxruntime")
    else:
        pip(venv, "install", "onnxruntime")

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
