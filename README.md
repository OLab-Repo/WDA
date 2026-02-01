# Wasserstein Distributional Attacks (WDA)

This repository contains the supplementary code for [*Tight Robustness Certificates and Wasserstein Distributional Attacks for Deep Neural Networks*](https://arxiv.org/abs/2510.10000).

## Overview

- `wda.py` - core WDA routines.
- `run_wda_attack.py` - Main experiment script. WDA plus our combined WDA + Adaptive Auto Attack variant (A³‑WDA).
- `adaptive_auto_attack/` - Code for A³‑WDA.
- `utils.py`: Shared utilities.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download weights on first use by running any of the scripts below; RobustBench handles caching automatically.

## Quick Start

Evaluate WDA on a RobustBench model:

```bash
python run_wda_attack.py \
  --model_name Sehwag2021Proxy \
  --dataset cifar10 \
  --eps 8/255 \
  --step_size_coeff 0.02 \
  --topk 10 \
  --wda_fixed
```

Evaluate our WDA++ method:

```bash
python run_wda_attack.py \
  --model_name Sehwag2021Proxy \
  --dataset cifar10 \
  --eps 8/255 \
  --step_size_coeff 0.02 \
  --topk 10 \
  --wda_pp
```

Evaluate our A³‑WDA method:

```bash
python run_wda_attack.py \
  --model_name Sehwag2021Proxy \
  --dataset cifar10 \
  --eps 8/255 \
  --step_size_coeff 0.02 \
  --topk 10 \
  --aaa
```

## Citation

Please cite the accompanying arXiv preprint when using this code:

```bib
@article{le2025tight,
  title={Tight Robustness Certificates and Wasserstein Distributional Attacks for Deep Neural Networks},
  author={Le, Bach C and Dao, Tung V and Nguyen, Binh T and Chu, Hong},
  journal={arXiv preprint arXiv:2510.10000},
  year={2025}
}
```

## License

This project is released under the MIT License; see `LICENSE` for details. Third-party components remain under their respective licences.