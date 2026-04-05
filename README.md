# Reproducing TPCRP: Active Learning on a Budget

Reproduction of the TPCRP algorithm from Hacohen et al., *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets* (ICML 2022), on CIFAR-10.

## What this does

- Trains a SimCLR encoder (ResNet-18, 500 epochs) on CIFAR-10 to get 512-d embeddings
- Implements TPCRP selection: K-Means clustering + KNN typicality scoring
- Compares against Random, KMeans, and Furthest First baselines at budgets B=10 and B=50
- Evaluates under Framework 1: ResNet-18 trained from scratch on the selected subset
- Proposes a modification (Task 3): centroid-proximity typicality instead of KNN

## Files

- `tpcrp_cifar10.ipynb` — full implementation (SimCLR training, TPCRP, baselines, experiments, Task 3)
- `simclr.pth` — pretrained SimCLR checkpoint (500 epochs)

## How to run

Tested on Google Colab with GPU runtime.

```bash
pip install torch torchvision scikit-learn matplotlib tqdm
```

Open `tpcrp_cifar10.ipynb` and run all cells. SimCLR training takes ~3 hours on a T4 GPU. If `simclr.pth` exists in the working directory, training is skipped and the checkpoint is loaded.

## Results

| Method | B=10 | B=50 |
|--------|------|------|
| TPCRP | .132 +/- .011 | .226 +/- .011 |
| Random | .149 +/- .016 | .196 +/- .016 |
| KMeans | .176 +/- .019 | .231 +/- .018 |
| FF | .147 +/- .016 | .178 +/- .013 |

10 repeats, Framework 1 (ResNet-18 from scratch, 200 epochs).
