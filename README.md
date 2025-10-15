# Flash-Attention 3 Wheels

**Pre-built wheels that erase CUDA / PyTorch compatibility headaches.**

[![Build Flash-Attention 3 Wheels](https://github.com/windreamer/flash-attention3-wheels/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/windreamer/flash-attention3-wheels/actions/workflows/build_wheels.yml)


## Quick start

Pick the line that matches your setup (change `cu128` / `torch280` if needed):

```bash
# CUDA 12.8 + PyTorch 2.8.0
pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
```

## How to pick the right index

Visit the GitHub Pages site (`https://windreamer.github.io/flash-attention3-wheels`) and choose the link that matches:

* CUDA 12.6 → `cu126_torch...`  
* CUDA 12.8 → `cu128_torch...`  
* CUDA 12.9 → `cu129_torch...`  

Each page shows the one-liner you need.

## When are wheels updated?

* **Weekly**, every Sunday at 22:00 UTC  
* **On demand**, by triggering the workflow manually if you need a fresher build

Releases are tagged with the build date (`2025.10.15`) so you always know how fresh your wheel is.

## License

The build scripts and index generator are Apache-2.0.  
