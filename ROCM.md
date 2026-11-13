# TripoSG on AMD GPUs (ROCm, Linux and Windows)

This branch runs TripoSG without its CUDA-only dependencies, so it works on AMD GPUs with ROCm PyTorch (and on
any other PyTorch device). Nothing needs compiling.

## What changed

| Upstream dependency | Here |
| --- | --- |
| `diso` (CUDA dual marching cubes) for the flash decoder | When `diso` is missing, cells the sparse refinement never decodes take the trilinearly upsampled value of the coarser level, and scikit-image's marching cubes runs on the complete grid. Same topology as upstream's dense path (one body, no open boundary edges); 1.3 s at 512³. |
| `torch_cluster.fps` (VAE point-cloud encoding only) | Plain PyTorch farthest point sampling when `torch_cluster` is missing. |
| older diffusers | Works with current diffusers (LayerNorm moved, custom scheduler module imported before loading). |

## Install

Install a ROCm build of PyTorch first (Linux: `pip install torch torchvision --index-url
https://download.pytorch.org/whl/rocm7.0`; Windows: AMD's ROCm wheels, or ComfyUI's Windows portable for AMD), then
the rest without letting pip replace it:

```
pip install diffusers transformers einops huggingface_hub opencv-python-headless trimesh omegaconf scikit-image peft jaxtyping typeguard pymeshlab
PYTHONPATH=. python scripts/inference_triposg.py --image-input image.png --output-path out.glb
```

## Faster inference

Measured on a Radeon PRO W7900 (gfx1100), image to mesh:

| Setting | Time | Faces |
| --- | ---: | ---: |
| defaults (50 steps, octree depth 9 = 512³) | 37 s | 1.14 M |
| + SageAttention ([SageAttention-rocm](https://github.com/TwelveNights/SageAttention-rocm)) | 30 s | 1.15 M |
| + 25 steps | 23 s | 1.16 M |
| + `flash_octree_depth=8` (256³) | **14 s** | 288 K |

Radeon RX 7800 XT 16 GB, Windows 11, same optimized settings: 15.9-18.9 s per object.

Depth 8 is as close to the 50-step, 512³ result as depth 9 (mean surface distance 0.07% of the bounding-box
diagonal for both) with a quarter of the faces, which is plenty for game assets. Pass `num_inference_steps=25,
flash_octree_depth=8` to the pipeline, and with SageAttention installed call
`from triposg.rocm_attention import use_sage_attention; use_sage_attention()` before running it.

Note: the background remover the scripts use (BRIA RMBG-1.4) is licensed for non-commercial use only; for commercial
assets, cut the background out yourself or use another matting model.
