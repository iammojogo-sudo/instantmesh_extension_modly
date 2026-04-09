"""
InstantMesh — Modly extension generator
Reference: https://github.com/TencentARC/InstantMesh

Pipeline:
  1. Remove background with rembg
  2. Zero123++ diffusion → 6 consistent multi-view images
  3. InstantMesh LRM reconstruction → FlexiCubes mesh
  4. Export GLB
"""
import io
import os
import sys
import random
import time
import uuid
import zipfile
import threading
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID    = "TencentARC/InstantMesh"
_GITHUB_ZIP    = "https://github.com/TencentARC/InstantMesh/archive/refs/heads/main.zip"
_ZERO123_REPO  = "sudo-ai/zero123plus-v1.2"

_CKPT_FILES = {
    "instant-mesh-large": "instant_mesh_large.ckpt",
    "instant-mesh-base":  "instant_mesh_base.ckpt",
}
_CONFIG_FILES = {
    "instant-mesh-large": "instant-mesh-large.yaml",
    "instant-mesh-base":  "instant-mesh-base.yaml",
}


class InstantMeshGenerator(BaseGenerator):
    MODEL_ID     = "instantmesh"
    DISPLAY_NAME = "InstantMesh"
    VRAM_GB      = 6

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        marker = self.model_dir / "diffusion_pytorch_model.bin"
        return marker.exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_source()

        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from huggingface_hub import hf_hub_download

        device = "cuda" if torch.cuda.is_available() else "cpu"
        src    = self._src_dir()

        print("[InstantMeshGenerator] Loading Zero123++ diffusion pipeline…")
        pipeline = DiffusionPipeline.from_pretrained(
            _ZERO123_REPO,
            custom_pipeline=str(src / "zero123plus"),
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        unet_path = self.model_dir / "diffusion_pytorch_model.bin"
        state_dict = torch.load(str(unet_path), map_location="cpu")
        pipeline.unet.load_state_dict(state_dict, strict=True)
        pipeline = pipeline.to(device)

        self._pipeline = pipeline
        self._device   = device
        self._src      = src
        self._model    = pipeline  # satisfies BaseGenerator.unload()
        print(f"[InstantMeshGenerator] Loaded Zero123++ on {device}.")

    def _load_recon(self, variant: str):
        """Lazy-load the reconstruction model for a given variant."""
        import torch
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download

        ckpt_filename = _CKPT_FILES[variant]
        cfg_filename  = _CONFIG_FILES[variant]

        ckpt_path = self.model_dir / ckpt_filename
        if not ckpt_path.exists():
            print(f"[InstantMeshGenerator] Downloading {ckpt_filename}…")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=ckpt_filename,
                repo_type="model",
                local_dir=str(self.model_dir),
            )

        cfg_path = self._src / "configs" / cfg_filename
        config   = OmegaConf.load(str(cfg_path))
        infer_cfg = config.infer_config

        sys.path.insert(0, str(self._src))
        from src.utils.train_util import instantiate_from_config

        model_config = OmegaConf.load(str(self._src / "configs" / cfg_filename))
        recon_model  = instantiate_from_config(model_config.model_config)
        state = torch.load(str(ckpt_path), map_location="cpu")
        state = {
            k[14:]: v
            for k, v in state["state_dict"].items()
            if k.startswith("lrm_generator.") and "source_camera" not in k
        }
        recon_model.load_state_dict(state, strict=False)
        recon_model = recon_model.to(self._device)
        recon_model.eval()
        return recon_model, infer_cfg

    def unload(self) -> None:
        super().unload()
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch
        import numpy as np
        from einops import rearrange
        from PIL import Image

        variant       = params.get("model_variant", "instant-mesh-large")
        diff_steps    = int(params.get("diffusion_steps", 75))
        export_texmap = bool(params.get("export_texmap", True))
        seed          = int(params.get("seed", 42))

        # -- Background removal -----------------------------------------
        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        # -- Multi-view diffusion ---------------------------------------
        self._report(progress_cb, 10, "Generating 6 multi-view images…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 55, "Generating multi-view images…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.no_grad():
                mv_result = self._pipeline(
                    image,
                    num_inference_steps=diff_steps,
                    generator=generator,
                ).images[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # mv_result is a 960×640 grid of 6 views (3 rows × 2 cols of 320×320)
        mv_np = np.asarray(mv_result, dtype=np.float32) / 255.0
        mv_tensor = torch.from_numpy(mv_np).permute(2, 0, 1).float()
        mv_views  = rearrange(mv_tensor, "c (n h) (m w) -> (n m) c h w", n=3, m=2)

        # -- Reconstruction ---------------------------------------------
        self._report(progress_cb, 58, "Loading reconstruction model…")
        recon_model, infer_cfg = self._load_recon(variant)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 65, "Reconstructing 3D mesh…")
        stop_evt2 = threading.Event()
        if progress_cb:
            t2 = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 65, 92, "Reconstructing mesh…", stop_evt2),
                daemon=True,
            )
            t2.start()

        try:
            with torch.no_grad():
                # Build camera poses (same as official run.py)
                input_cameras = self._get_zero123plus_input_cameras(
                    batch_size=1, radius=4.0
                ).to(self._device)

                planes = recon_model.forward_planes(
                    mv_views.unsqueeze(0).to(self._device),
                    input_cameras,
                )

                mesh_out = recon_model.extract_mesh(
                    planes,
                    use_texture_map=export_texmap,
                    **infer_cfg,
                )
        finally:
            stop_evt2.set()

        self._check_cancelled(cancel_event)

        # -- Export GLB ------------------------------------------------
        self._report(progress_cb, 94, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name    = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        out_path = self.outputs_dir / name

        # mesh_out is (vertices, faces, vertex_colors) or (vertices, faces, uvs, texture)
        self._export_glb(mesh_out, export_texmap, str(out_path))

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        try:
            result = rembg.remove(img)
        except Exception:
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            result  = rembg.remove(img, session=session)
        # White background composite, resize to 320×320 for Zero123++
        bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
        bg.paste(result, mask=result.split()[3])
        return bg.convert("RGB").resize((320, 320), Image.LANCZOS)

    def _get_zero123plus_input_cameras(self, batch_size: int = 1, radius: float = 4.0):
        """Build the 6 fixed camera poses that Zero123++ v1.2 uses."""
        import torch
        import numpy as np

        # elevations and azimuths matching Zero123++ v1.2 fixed poses
        elevations = torch.tensor([20.0, -10.0, 20.0, -10.0, 20.0, -10.0])
        azimuths   = torch.tensor([30.0, 90.0, 150.0, 210.0, 270.0, 330.0])

        # Use the source helper if available, else fall back to a simple impl
        try:
            sys.path.insert(0, str(self._src))
            from src.utils.camera_util import get_zero123plus_input_cameras
            return get_zero123plus_input_cameras(batch_size, radius)
        except ImportError:
            pass

        # Fallback: build camera extrinsics manually
        c2ws = []
        for elev, azim in zip(elevations.tolist(), azimuths.tolist()):
            elev_r = np.radians(elev)
            azim_r = np.radians(azim)
            x = radius * np.cos(elev_r) * np.cos(azim_r)
            y = radius * np.cos(elev_r) * np.sin(azim_r)
            z = radius * np.sin(elev_r)
            eye    = np.array([x, y, z])
            center = np.zeros(3)
            up     = np.array([0, 0, 1])
            f  = center - eye; f /= np.linalg.norm(f)
            r  = np.cross(f, up); r /= np.linalg.norm(r)
            u  = np.cross(r, f)
            c2w = np.eye(4)
            c2w[:3, 0] = r; c2w[:3, 1] = u; c2w[:3, 2] = -f; c2w[:3, 3] = eye
            c2ws.append(c2w)

        c2ws  = torch.tensor(np.stack(c2ws), dtype=torch.float32)
        # Simple camera bundle: [c2w_flat(16), focal(1), pp(2)] — adjust to recon model API
        return c2ws.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _export_glb(self, mesh_out, use_texture_map: bool, path: str) -> None:
        import trimesh
        import numpy as np

        if use_texture_map:
            vertices, faces, uvs, texture = mesh_out
            v_np = vertices.cpu().numpy()
            f_np = faces.cpu().numpy()
            uv_np = uvs.cpu().numpy()
            tex_np = (texture.cpu().numpy() * 255).astype(np.uint8)

            mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
            tex_img = Image.fromarray(tex_np)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tex_img.save(tf.name)
                tex_path = tf.name
            try:
                material = trimesh.visual.texture.SimpleMaterial(
                    image=Image.open(tex_path)
                )
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_np, material=material
                )
            finally:
                os.unlink(tex_path)
        else:
            vertices, faces, vertex_colors = mesh_out
            v_np  = vertices.cpu().numpy()
            f_np  = faces.cpu().numpy()
            vc_np = (vertex_colors.cpu().numpy() * 255).astype(np.uint8)
            mesh  = trimesh.Trimesh(
                vertices=v_np, faces=f_np,
                vertex_colors=vc_np, process=False
            )

        mesh.export(path)

    def _src_dir(self) -> Path:
        return self.model_dir / "_src" / "InstantMesh-main"

    def _ensure_source(self) -> None:
        src = self._src_dir()
        if (src / "src").exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
        self._download_source(src.parent)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

    def _download_source(self, dest: Path) -> None:
        import urllib.request
        dest.mkdir(parents=True, exist_ok=True)
        print("[InstantMeshGenerator] Downloading InstantMesh source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print("[InstantMeshGenerator] Extracting source…")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                rel    = member  # keep full relative path including InstantMesh-main/
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        print(f"[InstantMeshGenerator] Source extracted to {dest}.")

    def _download_weights(self) -> None:
        from huggingface_hub import hf_hub_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[InstantMeshGenerator] Downloading UNet weights…")
        hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
            local_dir=str(self.model_dir),
        )
        print("[InstantMeshGenerator] UNet weights downloaded.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "model_variant",
                "label":   "Quality",
                "type":    "select",
                "default": "instant-mesh-large",
                "options": [
                    {"value": "instant-mesh-large", "label": "Large (best quality)"},
                    {"value": "instant-mesh-base",  "label": "Base (faster)"},
                ],
                "tooltip": "Large gives better geometry. Base is faster.",
            },
            {
                "id":      "diffusion_steps",
                "label":   "Diffusion Steps",
                "type":    "select",
                "default": 75,
                "options": [
                    {"value": 30, "label": "Fast (30)"},
                    {"value": 75, "label": "Balanced (75)"},
                ],
                "tooltip": "Steps for the Zero123++ multi-view generation stage.",
            },
            {
                "id":      "export_texmap",
                "label":   "Texture Map",
                "type":    "bool",
                "default": True,
                "tooltip": "UV texture map. Disable for vertex-color only (faster).",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": 42,
                "min":     0,
                "max":     4294967295,
                "tooltip": "Change if result is unsatisfying.",
            },
        ]
