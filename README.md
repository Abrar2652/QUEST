# QUEST: Structural Priors and Confidence-Weighted Graph Regularization for Uncertain Knowledge Graph Completion



## Repository layout

```
QUEST/
├── quest_run.py                  # Training script
├── config/{nl27k,cn15k}/*.yaml   # QUEST + ablation configs
├── src/unKR/                     # ssCDL codebase
│   ├── model/UKGModel/
│   │   ├── SSCDL.py              
│   │   └── SSCDL_QUEST.py        # + spectral init + graph reg
│   └── lit_model/
│       ├── ssCDLLit.py           # ssCDL LitModel
│       └── ssCDLLit_QUEST.py     # + pre-PCDG-only smoothness loss
├── dataset/{nl27k,cn15k}/        # TSV triples (h r t s)
├── logs_*.txt                    # Training logs (5 runs)
├── output/                       # Checkpoints (best-MSE, best-MAE, best-WMRR)
└── make_figure{2,3,4}.py         # Figure generation scripts

```

## Quick start

```bash
cd /nas/home/jahin/QUEST

# Run QUEST on NL27k (GPU 0)
PYTORCH_NVML_BASED_CUDA_CHECK=0 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python quest_run.py --gpu 0 \
    --load_config --config_path config/nl27k/QUEST_nl27k.yaml

# Run QUEST on CN15k (GPU 1)
PYTORCH_NVML_BASED_CUDA_CHECK=0 CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python quest_run.py --gpu 0 \
    --load_config --config_path config/cn15k/QUEST_cn15k.yaml
```

## Ablation configurations

Four additional configs are provided under `config/*/` for the ablation
study:

| File | Change |
|------|--------|
| `QUEST_{nl27k,cn15k}_ablation_noSpec.yaml` | `use_spectral_init: false` |
| `QUEST_{nl27k,cn15k}_ablation_noReg.yaml`  | `lambda_smooth: 0.0` |

## Figures

- `figure2_training_dynamics.pdf` — per-epoch validation metrics for all
  three variants on both datasets; exposes the PCDG boundary instability
- `figure3_ablation_bars.pdf` — per-metric test-set results with %
  delta vs. ssCDL annotations
- `figure4_embedding_viz.pdf` — t-SNE of Xavier vs. spectral initialization
  (coloured by log-degree), showing spectral init already encodes graph
  communities before training


## Datasets

| Dataset | Entities | Relations | Train/Val/Test | Source |
|---------|---------:|----------:|---------------:|--------|
| NL27k   | 27,221 | 417 | 149k / 12k / 14k | NELL |
| CN15k   | 15,000 | 36  | 205k / 16k / 19k | ConceptNet |

85% / 7% / 8% split (same as UKGE and ssCDL).

## Dependencies

- Python 3.10, PyTorch 2.4.1+cu121, pytorch-lightning 1.9.5
- scipy, scikit-learn, numpy, matplotlib (for figures)
- tqdm, gensim, tensorboard, PyYAML


