# Setting up 2DGS + leaf picking on a new Ubuntu machine

Reproduces the `plant_10-16` pipeline from the COLMAP export onwards (2DGS training, depth maps, TSDF mesh,
SAM3 leaf masks, leaf tracking, surface picker) on a machine that has no gsplat.

Copied from the working machine (Ubuntu 26.04, RTX 5060 Laptop, driver 580):

| Piece | Version |
|---|---|
| Python | 3.11 (conda) |
| PyTorch | 2.11.0 + cu128, torchvision 0.26.0 |
| CUDA compiler for building gsplat | conda-forge `cuda-nvcc` 12.9, conda `gcc/gxx` 14 |
| gsplat | 1.5.3, upstream commit `937e2991`, built from source |
| SAM3 | `facebookresearch/sam3` commit `eef6839`, editable install |
| Open3D | 0.20.0 in the conda envs; **0.19.0** in a separate venv for the TSDF mesh (0.20 returns a mesh with 0 triangles) |
| numpy | 1.26.4 (gsplat examples and SAM3 both need numpy < 2) |

## 0. Prerequisites

- An NVIDIA GPU and driver **≥ 570** (needed for CUDA 12.8 wheels). Check with `nvidia-smi`.
  RTX 50-series (Blackwell) cards *require* the cu128 wheels; older cards (RTX 20/30/40) work with them too.
- No system CUDA toolkit is needed: the compiler comes from conda in step 2.
- Git, and ~5 GB free for the environments plus the data in step 5.

```bash
sudo apt update && sudo apt install -y git build-essential
# Miniconda, if `conda` is not installed:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && exec bash
```

## 1. Code

The scripts look for the repo at `~/Projects/repas-vision`. If you clone it elsewhere, `export REPAS_VISION=/path/to/repas-vision`.

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone <repas-vision remote> repas-vision
git clone https://github.com/nerfstudio-project/gsplat.git
git clone https://github.com/facebookresearch/sam3.git
cd gsplat && git checkout 937e2991 && git submodule update --init --recursive && cd ..
cd sam3  && git checkout eef6839 && cd ..
```

The custom 2DGS scripts are not part of gsplat. They are kept in this repo at `femto_bolt_code/gsplat_examples/`
and have to sit next to the stock trainer, because they import `simple_trainer_2dgs` and `datasets.colmap`:

```bash
cp ~/Projects/repas-vision/femto_bolt_code/gsplat_examples/*.py ~/Projects/gsplat/examples/
```

| Script | What it does |
|---|---|
| `simple_trainer_2dgs_masked.py` | 2DGS trainer used for the main run (normal loss only inside the object mask, held-out split from `heldout_views.json`) |
| `simple_trainer_2dgs_tofdepth.py` | Optional: same trainer plus a dense ToF depth loss (the λ comparison runs) |
| `diag_depth.py` | `export` stage writes the per-view depth / alpha / normal maps and `cameras.json` that every later step reads |
| `tsdf_mesh_2dgs.py` | TSDF mesh from those maps (runs in the Open3D 0.19 venv) |

## 2. `gsplat` env: training and map export

```bash
conda create -n gsplat python=3.11 -y
conda activate gsplat

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA compiler + headers and a gcc that nvcc 12.9 accepts (the system gcc may be too new)
conda install -y -c conda-forge cuda-version=12.9 cuda-nvcc cuda-cudart-dev cuda-libraries-dev \
      gcc_linux-64=14 gxx_linux-64=14
pip install ninja

# Build gsplat from source (several minutes). --no-build-isolation makes it use the torch above.
cd ~/Projects/gsplat
pip install --no-build-isolation .

# Example dependencies (pycolmap, fused-ssim, fused-bilagrid are compiled too)
cd examples
pip install --no-build-isolation -r requirements.txt
pip install "numpy<2" open3d==0.20.0 scipy
```

If the build cannot find `nvcc` or picks the wrong GPU architecture:

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch;print("%d.%d"%torch.cuda.get_device_capability())')"
pip install --no-build-isolation --force-reinstall --no-deps ~/Projects/gsplat
```

Check:

```bash
python -c "import torch, gsplat; from gsplat.rendering import rasterization_2dgs; \
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), gsplat.__version__)"
```

Expect `True`, your GPU's name and `1.5.3`. Run it from outside `~/Projects/gsplat` so the installed build is imported,
not the source folder.

## 3. `repas` env: SAM3 masks, leaf tracking, picker

A clone of `gsplat` plus SAM3. Used for `sam3_leaf_masks.py`, `leaf_track.py`, `surface_pick.py` and
`surface_pick_viewer.py`, and for the camera scripts in this repo.

```bash
conda create -n repas --clone gsplat -y
conda activate repas
pip install -e ~/Projects/sam3
pip install einops pycocotools "numpy<2"     # imported by sam3 but not declared
pip install pupil-apriltags                  # tag scripts (tag_pose_pick.py etc.)
```

SAM3 weights: copy `sam3.pt` to `~/Projects/repas-vision/sam3_weights/sam3.pt` (the folder is git-ignored).
They come from the gated Hugging Face repo `facebook/sam3`, or copy them from the old machine.

## 4. Open3D 0.19 venv: TSDF mesh only

Use the conda Python 3.11 so that an Open3D 0.19 wheel exists (the system Python on new Ubuntu is too new for it):

```bash
~/miniconda3/envs/gsplat/bin/python -m venv ~/.venvs/repas_open3d
~/.venvs/repas_open3d/bin/pip install open3d==0.19.0 "numpy<2"
```

## 5. Data to copy from the old machine

`captures/` is git-ignored. Copy these folders into the same place under `~/Projects/repas-vision/captures/femto_bolt/`
(about 2.1 GB, without old results):

| Folder | Size | Used by |
|---|---|---|
| `plant_10-16_clean55/` (skip `results/` to retrain, or copy it to keep the trained runs) | 0.5 GB (+0.5 GB results) | training, export, TSDF |
| `plant_10-16_deg_5_speed_5/` (raw capture: colour, depth, `camera_params.json`, `manifest.csv`, `mask/`, `tag_pick/`, `leaf_masks/`) | 1.2 GB | leaf masks, picker |
| `plant_10-16_tof_clean_obl55/` | 89 MB | picker (cleaned ToF pixels), ToF-depth trainer |
| `plant_10-16/results_10-16/plant_10-16_side/diag/tof_cache/` | 254 MB | TSDF check, picker |

```bash
OLD=user@old-machine
for d in plant_10-16_clean55 plant_10-16_deg_5_speed_5 plant_10-16_tof_clean_obl55 \
         plant_10-16/results_10-16/plant_10-16_side/diag/tof_cache; do
  mkdir -p ~/Projects/repas-vision/captures/femto_bolt/$(dirname $d)
  rsync -a --info=progress2 $OLD:Projects/repas-vision/captures/femto_bolt/$d/ \
        ~/Projects/repas-vision/captures/femto_bolt/$d/
done
```

## 6. Running the pipeline

```bash
CAP=~/Projects/repas-vision/captures/femto_bolt
DATA=$CAP/plant_10-16_clean55
RUN=$DATA/results/2dgs_7k
RAW=$CAP/plant_10-16_deg_5_speed_5
CLEAN=$CAP/plant_10-16_tof_clean_obl55
TOF=$CAP/plant_10-16/results_10-16/plant_10-16_side/diag/tof_cache
S=~/Projects/repas-vision/femto_bolt_code/3dr_scripts
```

**a. Train 2DGS** (`gsplat` env, about 4.5 min of training on the RTX 5060 Laptop, 0.9 GB GPU memory). These flags reproduce `2dgs_7k/cfg.json`:

```bash
conda activate gsplat && cd ~/Projects/gsplat/examples
python simple_trainer_2dgs_masked.py --data-dir $DATA --result-dir $RUN \
    --data-factor 1 --max-steps 7000 --eval-steps 7000 --save-steps 7000 \
    --near-plane 0.01 --far-plane 1e10 \
    --normal-loss --normal-start-iter 1633 --dist-loss --dist-start-iter 700 --disable-viewer
```

**b. Export depth / alpha / normal maps** (`gsplat` env), writes `$RUN/maps/`:

```bash
python diag_depth.py export --model 2dgs --run_dir $RUN --data_dir $DATA --raw_dir $RAW
```

**c. TSDF mesh** (Open3D 0.19 venv), writes `$RUN/mesh/tsdf_mesh_metres.ply`:

```bash
~/.venvs/repas_open3d/bin/python tsdf_mesh_2dgs.py --run_dir $RUN --data_dir $DATA --tof_cache $TOF
```

**d. SAM3 leaf masks** (`repas` env), writes `$RAW/leaf_masks/`. Skip if you copied that folder:

```bash
conda activate repas && cd $S
python sam3_leaf_masks.py $RAW --cameras $RUN/maps/cameras.json
```

**e. Leaf tracking** (`repas` env), writes `$RUN/leaf_track/`:

```bash
python leaf_track.py --run $RUN --masks $RAW/leaf_masks
```

**f. Picker** (`repas` env). The defaults point at these plant_10-16 folders, so on the same layout no flags are needed:

```bash
python surface_pick.py --selftest
python surface_pick_viewer.py --run $RUN --raw $RAW --tof-cache $TOF --clean-dir $CLEAN
# optional comparison with a ToF-depth-trained run:
python surface_pick_viewer.py --depth-run $DATA/results/2dgs_7k_tof
```

On a Wayland desktop, if MeshLab opens a blank window, start it with `QT_QPA_PLATFORM=xcb meshlab ...`.

**Optional: ToF-depth 2DGS run** (`gsplat` env), only for the comparison mesh:

```bash
cd ~/Projects/gsplat/examples
python simple_trainer_2dgs_tofdepth.py --like $RUN/cfg.json --result_dir $DATA/results/2dgs_7k_tof \
    --tof_cache $TOF --clean_dir $CLEAN --raw_dir $RAW --tof_lambda 1.0
# then steps b and c with RUN=$DATA/results/2dgs_7k_tof
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | The nvcc is older than 12.8. Use the conda `cuda-nvcc` from step 2. |
| `unsupported GNU version! gcc versions later than 14 are not supported` | Install `gcc_linux-64=14 gxx_linux-64=14` in the env (step 2) and re-activate it. |
| `torch.cuda.is_available()` is False | The driver is too old for cu128 (needs ≥ 570). |
| `ModuleNotFoundError: einops` / `pycocotools` when importing sam3 | `pip install einops pycocotools` in `repas`. |
| TSDF report says 0 triangles | It ran under Open3D 0.20. Use `~/.venvs/repas_open3d/bin/python`. |
| `ModuleNotFoundError: flora_lib` | The repo isn't at `~/Projects/repas-vision`; set `REPAS_VISION`. |
| Numbers differ slightly from the old run | Expected: 2DGS training is not bit-for-bit deterministic across GPUs. |
