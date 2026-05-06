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
- NumPy
- SciPy
- scikit-learn
- networkx

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Data

Default datasets:

- Cora
- Citeseer
- Pubmed

Place raw data under:

```text
data/
  cora/
  citeseer/
  pubmed/
```

(If your code auto-downloads datasets, this step can be skipped.)

---

## 4. Quick Start

### 4.1 Node Classification

```bash
python train_node_cls.py \
  --dataset cora \
  --hidden 16 \
  --epochs 200 \
  --lr 0.01 \
  --seed 42
```

### 4.2 Link Prediction

```bash
python train_link_pred.py \
  --dataset cora \
  --hidden 128 \
  --out_dim 64 \
  --epochs 200 \
  --lr 0.005 \
  --seed 42
```

---

## 5. Leakage-Controlled Protocol (Link Prediction)

To avoid information leakage, we use the following protocol:

1. Split positive edges into train/val/test = 80%/5%/15% (fixed random seed).
2. Build **training graph** \(G_{train}=(V,E_{train})\) only.
3. Perform all structural preprocessing on \(G_{train}\) only:
   - line-graph construction \(L(G_{train})\)
   - Node2Vec on \(L(G_{train})\)
   - normalized operators \(\hat{A}, \hat{B}\)
4. Validation/test positive edges are **never** used in feature construction or message passing.
5. Negative samples are drawn from non-edges with 1:1 positive/negative ratio, excluding all known positives.
6. Use validation AUC for model selection; report test AUC once.
7. Use the same splits and negative samples for all baselines.
