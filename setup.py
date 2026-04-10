# -*- coding: utf-8 -*-
"""
InstantMesh - Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv, *args, extra_env=None):
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([str(pip_exe)] + list(args), check=True, env=env)


def find_cuda_home(venv):
    """
    Find CUDA home. Prefer system install, fall back to the CUDA libs
    bundled inside the PyTorch wheel (which pip installs into the venv).
    """
    # 1. Explicit env vars set by user
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return val

    # 2. Default system CUDA install location on Windows
    cuda_base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_base.exists():
        versions = sorted(cuda_base.iterdir(), reverse=True)
        for v in versions:
            if (v / "bin" / "nvcc.exe").exists():
                return str(v)

    # 3. Fall back to PyTorch's bundled CUDA (no system install needed)
    #    torch ships nvcc inside torch/bin on Windows cu* wheels
    is_win = platform.system() == "Windows"
    site_packages = venv / ("Lib/site-packages" if is_win else "lib/python*/site-packages")
    if is_win:
        torch_cuda = venv / "Lib" / "site-packages" / "torch"
    else:
        import glob
        matches = glob.glob(str(venv / "lib" / "python*" / "site-packages" / "torch"))
        torch_cuda = Path(matches[0]) if matches else None

    if torch_cuda and torch_cuda.exists():
        # torch bundles CUDA runtime under torch/lib and nvcc under torch/bin
        nvcc = torch_cuda / "bin" / "nvcc.exe" if is_win else torch_cuda / "bin" / "nvcc"
        if nvcc.exists():
            print("[setup] Using PyTorch bundled CUDA at: %s" % torch_cuda)
            return str(torch_cuda)

    return None


def setup(python_exe, ext_dir, gpu_sm):
    venv   = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d -> PyTorch 2.5.1 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers...")
    if gpu_sm >= 70:
        pip(venv, "install", "xformers==0.0.28.post3", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Windows Triton
    # ------------------------------------------------------------------ #
    if is_win:
        print("[setup] Installing Windows prebuilt triton...")
        triton_whl = (
            "https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/"
            "triton-2.0.0-cp310-cp310-win_amd64.whl"
        )
        try:
            pip(venv, "install", triton_whl)
        except subprocess.CalledProcessError:
            print("[setup] Triton whl install failed - skipping.")

    # ------------------------------------------------------------------ #
    # Pillow first
    # ------------------------------------------------------------------ #
    print("[setup] Installing Pillow...")
    pip(venv, "install", "Pillow>=9.0.0")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    core_pkgs = [
        "diffusers==0.27.2",
        "transformers==4.40.2",
        "accelerate",
        "huggingface_hub==0.23.5",
        "omegaconf",
        "einops",
        "numpy",
        "scipy",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "opencv-python-headless",
        "tqdm",
        "safetensors",
    ]
    for pkg in core_pkgs:
        print("[setup] Installing %s ..." % pkg)
        pip(venv, "install", pkg)

    # ------------------------------------------------------------------ #
    # nvdiffrast - built from source
    # ------------------------------------------------------------------ #
    print("[setup] Installing nvdiffrast build deps...")
    pip(venv, "install", "setuptools", "wheel", "ninja")

    cuda_home = find_cuda_home(venv)
    if cuda_home:
        print("[setup] CUDA_HOME resolved to: %s" % cuda_home)
    else:
        print("[setup] WARNING: CUDA_HOME not found. nvdiffrast build may fail.")

    build_env = {}
    if cuda_home:
        build_env["CUDA_HOME"] = cuda_home
        build_env["CUDA_PATH"] = cuda_home

    print("[setup] Installing nvdiffrast from GitHub...")
    pip(venv, "install", "git+https://github.com/NVlabs/nvdiffrast.git",
        "--no-build-isolation", extra_env=build_env)

    # ------------------------------------------------------------------ #
    # rembg
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg...")
    if gpu_sm >= 70:
        try:
            pip(venv, "install", "rembg[gpu]")
        except subprocess.CalledProcessError:
            pip(venv, "install", "rembg")
    else:
        pip(venv, "install", "rembg")

    # ------------------------------------------------------------------ #
    # onnxruntime
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            pip(venv, "install", "onnxruntime")
    else:
        pip(venv, "install", "onnxruntime")

    print("[setup] Done. Venv ready at: %s" % venv)


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
