# ILNT-GCN

Official implementation of **ILNT-GCN (Incorporating Line Graph Topology Feature in Graph Convolutional Networks)**.

## 1. Overview

ILNT-GCN is a lightweight edge-to-node fusion framework for graph representation learning:

- Use Node2Vec on line graph \(L(G)\) to obtain edge-side topology embeddings.
- Concatenate line-graph embeddings with edge features.
- Inject edge-side signals into node representations via normalized incidence aggregation in the first layer.
- Keep later layers as standard node-level GCN propagation.

Supported tasks:

- Node Classification
- Link Prediction

---

## 2. Environment

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric (PyG)
- NumPy, SciPy, scikit-learn, networkx

Install dependencies:

```bash
pip install -r requirements.txt
## 3. Data
Default datasets:

Cora
Citeseer
Pubmed
Place raw data under:

data/
  cora/
  citeseer/
  pubmed/
## 4. Quick Start
# 4.1 Node Classification
python train_node_cls.py \
  --dataset cora \
  --hidden 16 \
  --epochs 200 \
  --lr 0.01 \
  --seed 42
# 4.2 Link Prediction
python train_link_pred.py \
  --dataset cora \
  --hidden 128 \
  --out_dim 64 \
  --epochs 200 \
  --lr 0.005 \
  --seed 42
