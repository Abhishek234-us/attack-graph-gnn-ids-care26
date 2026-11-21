# Attack Graph based Intrusion Detection with GNNs (CARE 2026, Paper ID: CARE-26-122)

GNN models (GCN, GCN-EW, GAT, GraphSAGE+RF) and a traffic-only NN baseline that detect the
attacked privilege node in a 51-node MulVAL attack graph, using CIC-IDS2017 flow features.

## Notebooks
| File | Model |
|---|---|
| 1nn_multiseed.ipynb | NN (traffic only, baseline) |
| 2gcn_multiseed.ipynb | GCN |
| 3gcn_ew_multiseed.ipynb | GCN-EW (learnable edge weights) |
| 4gat_multiseed.ipynb | GAT |
| 5gsage_multiseed.ipynb | GraphSAGE + Random Forest |

## Evaluation notes
- Mean over 5 seeds; 60/20/20 stratified split.
- The MinMax scaler is fit on the training split only, and benign/attack flows are sampled at random
  (the original GNN-IDS code scales before the split and takes the first rows of the CSV).
- Precision/Recall/F1 are reported both for the attack class and macro-averaged (as in GNN-IDS).

## Reference
Z. Sun et al., "GNN-IDS: Graph Neural Network based Intrusion Detection System," ARES 2024.
