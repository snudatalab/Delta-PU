# Delta-PU: Dual-level Meta-reweighting for Positive-Unlabeled Graph Classification

This is the official implementation of **Delta-PU** (*Dual-level Meta-reweighting for Positive-Unlabeled Graph Classification*), accepted to WWW 2026.

---

## 📦 Requirements

We recommend using the following versions:

```bash
python==3.9.21
torch==2.7.0+cu118
torch-geometric==2.6.1
torchvision==0.22.0+cu118
torchaudio==2.7.0+cu118
scikit-learn==1.6.1
scipy==1.13.1
pandas==2.2.3
tqdm==4.67.1
```

---

## 📁 Code Structure

- `main.py`: Entry script for training the PU graph classifier.
- `models/gnn.py`: Implements the GIN-based model used for classification.
- `models/loss.py`: Contains PU loss functions (balanced CE, meta reweighting).
- `models/train.py`: Training loop with dual-level reweighting logic.
- `models/utils.py`: Utility functions.
- `models/data.py`: Loads datasets from TUDataset and processes them.

---

## 🧪 How to Run

You can reproduce the experimental results in the paper with the following commands:

```bash
python main.py --dataset MUTAG --loss dump
python main.py --dataset NCI1 --loss dump
python main.py --dataset NCI109 --loss dump
python main.py --dataset PROTEINS --loss dump
python main.py --dataset facebook_ct1 --loss dump
python main.py --dataset BAMultiShapes --loss dump
```

### Main Arguments
| Argument | Description                          |
|----------|--------------------------------------|
| `--dataset` | Dataset name (`MUTAG`, `NCI1`, etc.) |
| `--loss` | Use `dump` to enable Delta-PU        |
| `--gpu` | GPU index                            |
| `--epochs` | Number of training epochs            |
| `--lr` | Learning rate for training           |
| `--meta-lr` | Learning rate for meta reweighting   |
| `--batch-size` | Batch size for training              |

---

## 📊 Supported Datasets

| Dataset   | Description |
|-----------|-------------|
| MUTAG     | Mutagenicity prediction of compounds |
| NCI1      | Activity prediction for lung cancer cells |
| NCI109    | Activity prediction for ovarian cancer cells |
| PROTEINS  | Classify protein structures |
| facebook_ct1  | Classify rumor / truth |
| BAMultiShapes  | Synthetic data |

All datasets are automatically downloaded from [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).
