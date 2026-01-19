# ------------------------------------------------------------------------
# LDM-MC-SFYOLO: Latent Diffusion Based Target Augmentation
# This module implements the diffusion-based augmentation pipeline.
# ------------------------------------------------------------------------

import os
from pathlib import Path
import torch
import numpy as np
from typing import List, Optional, Tuple

# Lazy import diffusers to avoid hard dependency if not using LDM
_ldm_pipe = None


def _load_ldm_pipeline(model_id: str = "stabilityai/stable-diffusion-2-1",
                       device: Optional[torch.device] = None,
                       torch_dtype: Optional[torch.dtype] = torch.float16,
                       use_safetensors: bool = True):
    global _ldm_pipe
    if _ldm_pipe is not None:
        return _ldm_pipe
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except Exception as e:
        raise RuntimeError(
            "diffusers is required for LDM augmentation. Install with `pip install diffusers transformers accelerate safetensors`"
        ) from e

    # CPU needs float32, CUDA can use float16
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (device.type == 'cuda') else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=use_safetensors
    )
    pipe.safety_checker = None  # disable NSFW checker for speed and determinism
    pipe.requires_safety_checker = False
    pipe = pipe.to(device)
    # Try to enable xformers if available; fall back silently if not installed
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as _e:
            print(f"[WARN] xformers not available or incompatible; continuing without. Details: {_e}")
    # Only try cpu offload if accelerate is present and supported; ignore otherwise
    try:
        if device.type == 'cpu' and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
    except Exception:
        pass
    # Freeze all learnable params of core modules if present (pipeline itself is not nn.Module)
    for _mod_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        _m = getattr(pipe, _mod_name, None)
        if _m is None:
            continue
        try:
            for _p in _m.parameters():
                _p.requires_grad = False
        except Exception:
            pass
    _ldm_pipe = pipe
    return pipe


def _derive_cache_paths(paths: List[str], cache_root: Path, params_sig: str) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        try:
            # try to preserve substructure after 'images/'
            parts = Path(p).parts
            if 'images' in parts:
                idx = parts.index('images')
                rel = Path(*parts[idx+1:])
            else:
                rel = Path(Path(p).name)
            out.append(cache_root / params_sig / rel.with_suffix('.png'))
        except Exception:
            out.append(cache_root / params_sig / (Path(p).stem + '.png'))
    return out


def ldm_fog_augment(im_data_uint8: torch.Tensor,
                    prompt: str = "foggy city street, dense fog, haze, realistic",
                    negative_prompt: str = "clear sky, sunny, no fog, haze-free",
                    strength: float = 0.35,
                    guidance_scale: float = 5.0,
                    num_inference_steps: int = 25,
                    model_id: str = "stabilityai/stable-diffusion-2-1",
                    seed: Optional[int] = None,
                    device: Optional[torch.device] = None,
                    cache_paths: Optional[List[Path]] = None,
                    apply_mask: Optional[List[bool]] = None) -> torch.Tensor:
    """
    Apply Latent Diffusion (img2img) to add fog style to a batch of images.
    """
    assert im_data_uint8.dim() == 4, "Expected BCHW tensor"
    b, c, h, w = im_data_uint8.shape
    in_device = im_data_uint8.device
    pipe = _load_ldm_pipeline(model_id=model_id, device=device)

    # Convert to PIL list expected by diffusers
    if im_data_uint8.dtype != torch.uint8:
        im_uint8 = im_data_uint8.clamp(0, 255).to(torch.uint8)
    else:
        im_uint8 = im_data_uint8

    # Diffusers expects images in RGB HWC range 0..255 as PIL images
    imgs: List["PIL.Image.Image"] = []
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow is required for LDM augmentation. Install with `pip install pillow`." ) from e

    indices_to_run: List[int] = list(range(b))
    if apply_mask is not None:
        indices_to_run = [i for i, m in enumerate(apply_mask) if m]

    for i in indices_to_run:
        arr = im_uint8[i].detach().cpu().numpy().transpose(1, 2, 0)  
        arr = arr[..., ::-1]  
        imgs.append(Image.fromarray(arr))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    # Prepare output container
    out_tensor = torch.empty((b, 3, h, w), dtype=torch.uint8)
    
    # Try to load from cache
    if cache_paths is not None:
        for i, cp in enumerate(cache_paths):
            if cp is not None and cp.is_file():
                try:
                    from PIL import Image
                    arr = np.array(Image.open(cp).convert('RGB')).astype(np.uint8)
                    out_tensor[i] = torch.from_numpy(arr).permute(2, 0, 1)
                except Exception:
                    pass
    
    gen_indices: List[int] = []
    for i in range(b):
        need = True
        if apply_mask is not None and not apply_mask[i]:
            need = False
        if cache_paths is not None:
            cp = cache_paths[i]
            if cp is not None and cp.is_file():
                need = False
        if need:
            gen_indices.append(i)

    out_imgs: List["PIL.Image.Image"] = []
    if gen_indices:
        # [Fix] Handle dynamic prompts correctly
        if isinstance(prompt, list):
            if len(prompt) != b:
                print(f"[WARN] Prompt list length ({len(prompt)}) != Batch size ({b}). Using first prompt.")
                actual_prompts = [prompt[0]] * len(gen_indices)
            else:
                actual_prompts = [prompt[i] for i in gen_indices]
        else:
            actual_prompts = [prompt] * len(gen_indices)

        # [Fix] Handle dynamic negative prompts (Fixed variable name typo here)
        if isinstance(negative_prompt, list):
            if len(negative_prompt) != b:
                actual_negative_prompts = [negative_prompt[0]] * len(gen_indices)
            else:
                actual_negative_prompts = [negative_prompt[i] for i in gen_indices] # Fixed typo
        else:
            actual_negative_prompts = [negative_prompt] * len(gen_indices)

        with torch.no_grad():
            out = pipe(
                prompt=actual_prompts, 
                image=[imgs[indices_to_run.index(i)] for i in gen_indices],
                negative_prompt=actual_negative_prompts, 
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            )
        out_imgs = out.images
        
        # Save to cache if requested
        if cache_paths is not None:
            for i, img in zip(gen_indices, out_imgs):
                cp = cache_paths[i]
                try:
                    if cp is not None:
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        img.save(cp)
                except Exception:
                    pass
        
        # Fill tensor
        for i, img in zip(gen_indices, out_imgs):
            arr = np.array(img).astype(np.uint8)
            out_tensor[i] = torch.from_numpy(arr).permute(2, 0, 1)

    # For any remaining unset entries (pass-through or failed cache), use input
    for i in range(b):
        if (out_tensor[i] == 0).all():
            out_tensor[i] = im_uint8[i].detach().cpu()

    # Convert RGB->BGR and move back to original device
    out_tensor = out_tensor[:, [2, 1, 0], :, :]  # RGB->BGR
    return out_tensor.to(in_device).to(torch.float32)


def get_ldm_images(im_data_255: torch.Tensor, args) -> torch.Tensor:
    """
    Helper used by training loop to mimic TAM API.
    Expects images in 0..255 range (uint8 or float), returns 0..255 float.
    """
    # Optional per-image probability to apply LDM
    prob = float(getattr(args, 'ldm_prob', 1.0))
    b = im_data_255.shape[0]
    apply_mask = None
    if prob < 1.0:
        rng = np.random.default_rng()
        apply_mask = [bool(x) for x in (rng.random(b) < prob)]
    # Optional on-disk cache
    cache_dir = getattr(args, 'ldm_cache_dir', '') or ''
    cache_paths = None
    if cache_dir and hasattr(args, 'imgs_paths'):
        params_sig = f"s{getattr(args,'ldm_strength',0.35)}_g{getattr(args,'ldm_guidance_scale',5.0)}_n{getattr(args,'ldm_steps',25)}"
        cache_paths = _derive_cache_paths(list(getattr(args, 'imgs_paths', [])), Path(cache_dir), params_sig)
    return ldm_fog_augment(
        im_data_uint8=im_data_255,
        prompt=getattr(args, 'ldm_prompt', "foggy city street, dense fog, haze, realistic"),
        negative_prompt=getattr(args, 'ldm_negative_prompt', "clear sky, sunny, no fog, haze-free"),
        strength=float(getattr(args, 'ldm_strength', 0.35)),
        guidance_scale=float(getattr(args, 'ldm_guidance_scale', 5.0)),
        num_inference_steps=int(getattr(args, 'ldm_steps', 25)),
        model_id=getattr(args, 'ldm_model', "stabilityai/stable-diffusion-2-1"),
        seed=int(getattr(args, 'ldm_seed', -1)) if getattr(args, 'ldm_seed', None) not in (None, -1) else None,
        device=None,
        cache_paths=cache_paths,
        apply_mask=apply_mask
    )
