# ADGformer: Adaptive Dynamic Graph Transformer for Multivariate Time Series Forecasting

This repository contains the **core model architecture** of ADGformer.

> **Note:** The complete training pipeline, experiment scripts, and datasets will be released upon paper acceptance. Currently, only the model implementation is provided for reference.

## Overview

ADGformer tackles multivariate time series forecasting by adaptively learning dynamic inter-variable correlations and capturing multi-scale temporal dependencies. The model integrates four key components:

- **DCGL (Dynamic Correlation Graph Learning):** Learns time-varying adjacency matrices via learnable node embeddings, adaptively fusing a static Fourier-initialized correlation matrix with dynamic graph structures.
- **S-MHA (Spatial Multi-Head Attention):** Computes spatial attention scores to guide message passing between variables.
- **Weighted GAT (Graph Attention Network):** Performs spatial message passing on both dynamic and static graphs with S-MHA bias, fused via a learnable gating mechanism.
- **MTT (Multi-scale Temporal Transformer):** A hierarchical Transformer encoder with progressive patch merging to capture temporal dependencies at multiple scales.
- **GTU (Gated Temporal Unit):** Multi-kernel temporal convolution with gating for capturing temporal patterns at different receptive fields.

## Code Structure

```
├── models/
│   └── ADGformer.py              # Model entry point + Fourier-based static graph initialization
├── layers/
│   ├── ADGFormer_framework.py    # Core framework: DCGL, S-MHA, GAT, MTT, GTU, Predictor
│   ├── ADGFormer_layers.py       # Building blocks: positional encoding, Transformer encoder, etc.
│   └── RevIN.py                  # Reversible Instance Normalization
└── requirements.txt              # Python dependencies
```

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- numpy
- pandas
- einops

## Quick Start

```python
import torch
import pandas as pd
from models.ADGformer import Model

# A minimal config namespace (replace with actual argparse in full release)
class Configs:
    def __init__(self):
        # Data
        self.root_path = "./data"
        self.data_path = "ETTh1.csv"
        self.data = "ETTh1"
        self.n_vars = 7
        self.seq_len = 96
        self.pred_len = 96

        # DCGL
        self.num_adj_matrices = 7
        self.numpoint_win = 24
        self.w_bias = 0
        self.d_graph = 30
        self.d_gcn = 1
        self.w_ratio = 0.5
        self.mp_layers = 2

        # GAT
        self.gat_layers = 1
        self.gat_dropout = 0.0
        self.bias_scale = 1.0
        self.gat_tau = 1.0

        # S-MHA
        self.s_mha_d_model = 64
        self.s_mha_heads = 4
        self.s_mha_d_k = None

        # GTU
        self.gtu_stride = 1

        # MTT
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 64
        self.dropout = 0.1
        self.attn_dropout = 0.0
        self.predictor_dropout = 0.2
        self.embed = "timeF"
        self.activation = "gelu"

        # Patching
        self.patch_len = 2
        self.stride = 2

        # RevIN
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Configs()
model = Model(configs).to(configs.device)

# Dummy input: [batch_size, seq_len, n_vars]
batch_size = 32
x = torch.randn(batch_size, configs.seq_len, configs.n_vars).to(configs.device)
time_index = torch.randint(0, configs.num_adj_matrices, (batch_size, configs.seq_len)).to(configs.device)
current_epoch = 10

# Forward pass
output = model(x, time_index, current_epoch)
print(f"Output shape: {output.shape}")  # [batch_size, pred_len, n_vars]
```

## License

This project is licensed under the MIT License.

## Citation

If you find this work helpful, please cite our paper:

```
@article{...}
```
