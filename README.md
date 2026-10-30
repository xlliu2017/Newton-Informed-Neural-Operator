# Newton-Informed Neural Operator for Computing Multiple Solutions of Nonlinear Partial Differential Equations

This repository contains the implementation accompanying the **Newton-Informed Neural Operator** study of nonlinear PDE systems with **multiple valid solution branches**. The codebase focuses on combining neural operators with Newton-inspired residual information so that the learned surrogate is not only accurate, but also better aligned with the structure of the governing equations.

- **Paper page:** https://openreview.net/forum?id=F9mNL6vR27
- **Primary use cases in this repository:** convex problems, non-convex problems, and Gray-Scott reaction-diffusion systems
- **Core implementation styles:** DeepONet-style models, MgNO baselines, and FNO baselines

> In practical terms, this repository is aimed at researchers who want to explore how operator learning can be made more reliable for nonlinear PDEs where traditional single-solution surrogates can struggle.

## Why this repository matters

Neural operators are attractive because they learn mappings between function spaces and can generalize across discretizations and forcing conditions. For nonlinear PDEs with **coexisting solutions**, however, learning a stable and physically meaningful surrogate is harder. This repository tackles that challenge by pairing operator learning with **Newton-style residual information**, making it a useful starting point for:

- multi-solution PDE learning,
- physics-aware operator learning,
- surrogate modeling for nonlinear systems,
- and benchmarking Newton-informed architectures against established neural operator baselines.

## Repository map

The repository is compact, script-driven, and centered around a few key entry points:

- `newton_single_solution.py`  
  Training entry point for the convex/single-solution setting.

- `newton.py`  
  Training entry point for the non-convex setting.

- `newton_multi_solution.py`  
  Training entry point for Gray-Scott and related multi-solution experiments.

- `training_cli.py`  
  Shared command-line and configuration builder used by the three training scripts. This keeps experiment configuration consistent and makes the training entry points easier to read.

- `models.py`  
  Core model definitions, including multigrid-inspired neural operator variants.

- `utilities3.py`  
  Data handling, losses, and training utilities.

- `GrayScott.py`  
  Gray-Scott data generation utilities.

- `baselines/`  
  Baseline implementations such as FNO, U-Net, and related models.

## Environment setup

The repository ships with a conda-style dependency lock file:

```bash
conda create --name nino --file requirements.txt
conda activate nino
```

If you already have a compatible environment, you can install from the same dependency list there as well.

## Data

Datasets are available from the shared drive linked below:

- https://drive.google.com/drive/folders/1E-q7niAIkgdaP0lF9zDVNQvNmNGXotQn?usp=sharing

For Gray-Scott data generation, run:

```bash
python GrayScott.py
```

The repository also includes sample arrays such as `A.npy` and `S.npy` for local experimentation.

## Training workflows

Before training, make sure the data paths inside the scripts point to your local dataset locations.

### 1) Convex problem

**Train with PDE loss**

```bash
python newton_single_solution.py \
  --model_type DeepONet \
  --num_channel_u 48 \
  --num_layer 4 \
  --num_channel_f 2 \
  --final_div_factor 50 \
  --weight_decay 1e-6 \
  --lr 1e-4 \
  --batch_size 50 \
  --epochs 1000 \
  --loss_type pde
```

**Train with L2 loss**

```bash
python newton_single_solution.py \
  --model_type DeepONet \
  --num_channel_u 48 \
  --num_layer 4 \
  --num_channel_f 2 \
  --final_div_factor 50 \
  --weight_decay 1e-6 \
  --lr 1e-4 \
  --batch_size 50 \
  --epochs 1000 \
  --loss_type l2
```

### 2) Non-convex problem

```bash
python newton.py \
  --model_type DeepONet \
  --num_channel_u 48 \
  --num_layer 4 \
  --num_channel_f 2 \
  --final_div_factor 50 \
  --weight_decay 1e-6 \
  --lr 1e-4 \
  --batch_size 50 \
  --epochs 1000 \
  --loss_type pde
```

### 3) Gray-Scott multi-solution problem

```bash
python newton_multi_solution.py \
  --model_type DeepONet \
  --num_channel_u 48 \
  --num_layer 4 \
  --num_channel_f 2 \
  --final_div_factor 50 \
  --weight_decay 1e-6 \
  --lr 1e-4 \
  --batch_size 50 \
  --epochs 1000 \
  --loss_type pde
```

## Shared CLI options

The three training scripts now share the same configuration flow through `training_cli.py`, so the common options behave consistently:

- `--data`
- `--model_type`
- `--epochs`
- `--batch_size`
- `--optimizer_type`
- `--lr`
- `--final_div_factor`
- `--weight_decay`
- `--loss_type`
- `--sample_x`
- `--sampling_rate`
- `--normalizer`
- `--normalizer_type`
- `--num_layer`
- `--num_channel_u`
- `--num_channel_f`
- `--num_iteration`
- `--padding_mode`
- `--last_layer`
- `--test`
- `--MODEL_PATH_LOAD`

Example:

```bash
python newton.py --help
```

## Validation

This repository includes lightweight script-based validation utilities:

```bash
python -m unittest test_training_cli.py
python test_laplacian.py
```

`test_training_cli.py` validates the shared experiment CLI configuration, while `test_laplacian.py` checks consistency between a Laplacian matrix formulation and its convolutional implementation.

## Practical notes

- `utilities3.py` contains most of the training/data utilities used by the experiment scripts.
- `models.py` contains the main neural operator architectures and baseline-compatible building blocks.
- `newton.py`, `newton_single_solution.py`, and `newton_multi_solution.py` are the fastest places to start if you want to reproduce experiments or adapt the training recipes to a new PDE family.

## Citation

If this repository is useful in your research, please cite the corresponding OpenReview paper:

```bibtex
@misc{newton_informed_neural_operator,
  title={Newton-Informed Neural Operator for Computing Multiple Solutions of Nonlinear Partial Differential Equations},
  howpublished={OpenReview},
  url={https://openreview.net/forum?id=F9mNL6vR27}
}
```
