# auto-static-lod

This repo generates automatically LOD and normal map of high resolution mesh
By using differentable rasterization, this repo can generate statically LOD and normal from high resolution mesh input.
These contain total two processes.
1. Create low resolution mesh of the high from template mesh.
2. Create normal map of low resolution mesh by difference between high and low

## Install
with conda
`window`
```
INSTALL.bat
conda create env --file environment.yml
```
`ubuntu`
```
bash INSTALL.sh
conda create env --file environment.yml
```

## Usage
will be updated
```python
conda activate lod
python train.py
```
