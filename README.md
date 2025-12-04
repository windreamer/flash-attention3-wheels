# Flash-Attention 3 Wheels

**Pre-built wheels that erase [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) installation headaches — now with Windows support! 🎉 **

[![Build Flash-Attention 3 Wheels](https://github.com/windreamer/flash-attention3-wheels/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/windreamer/flash-attention3-wheels/actions/workflows/build_wheels.yml)

> **🚀 Update: Windows Wheels Now Available!**
> We've successfully built Flash Attention 3 wheels for **Windows** (CUDA 12.8 only for now).
>
> **Upstream PR:** Windows compatibility fixes submitted to [Dao-AILab/flash-attention#2047](http://github.com/Dao-AILab/flash-attention/pull/2047)
>
> **Note:** Until the PR is merged, wheels are built from our fork: [windreamer/flash-attention@fix_windows_fa3](https://github.com/windreamer/flash-attention/tree/fix_windows_fa3)


## Quick start

Pick the line that matches your setup (change `cu128` / `torch280` if needed):

```bash
# CUDA 12.8 + PyTorch 2.8.0
pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
```

## How to pick the right index

Visit the [GitHub Pages site](https://windreamer.github.io/flash-attention3-wheels) and choose the link that matches:

* CUDA 13.0 → `cu130_torch...`
* CUDA 12.9 → `cu129_torch...`
* CUDA 12.8 → `cu128_torch...`
* CUDA 12.6 → `cu126_torch...`

Each page shows the one-liner you need.

## When are wheels updated?

* **Weekly**, every Sunday at 22:00 UTC
* **On demand**, by triggering the workflow manually if you need a fresher build

Releases are tagged with the build date (`2025.10.15`) so you always know how fresh your wheel is.

## License

The build scripts and index generator are Apache-2.0.
