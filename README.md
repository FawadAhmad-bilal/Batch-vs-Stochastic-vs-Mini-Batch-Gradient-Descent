# Batch vs Stochastic vs Mini-Batch Gradient Descent

A deep learning experiment comparing different gradient descent strategies on the **Diamonds dataset** using Keras/TensorFlow.

## Overview

This notebook trains a neural network regression model to predict diamond prices, using **mini-batch gradient descent** via the Adam optimizer. It explores the effect of `batch_size` on training behavior and visualizes loss and R² curves across epochs.

## Dataset

- **Source:** Seaborn's built-in `diamonds` dataset
- **Samples:** ~54,000 diamonds
- **Features (9):** `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`
- **Target:** `price` (USD)

> **Note:** Categorical columns (`cut`, `color`, `clarity`) were kept as-is (label-encoded implicitly by Seaborn). No OHE or StandardScaler was applied in the final run.

## Model Architecture

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 122 | ReLU |
| Dense | 122 | ReLU |
| Dense | 122 | ReLU |
| Dense | 122 | ReLU |
| Dense | 12 | ReLU |
| Dense | 12 | ReLU |
| Dense (output) | 1 | Linear |

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Metric:** R² Score
- **Epochs:** 28
- **Batch Size:** 56 (Mini-Batch)

## Gradient Descent Variants

| Type | Batch Size | Behavior |
|------|-----------|----------|
| **Batch GD** | Full dataset | Slow, stable updates |
| **Stochastic GD (SGD)** | 1 sample | Fast, noisy updates |
| **Mini-Batch GD** ✅ | Small fixed size (e.g. 56) | Balanced — fast + stable |

This notebook uses **mini-batch** (batch_size=56), which is the standard approach in deep learning.

## Results

| Metric | Value |
|--------|-------|
| R² Score | ~0.91+ |
| Loss Tracked | MSE (train + val) |

Training and validation loss/R² are plotted to visualize convergence.

## Requirements

```
tensorflow
keras
pandas
numpy
seaborn
scikit-learn
matplotlib
```

Install with:
```bash
pip install tensorflow pandas numpy seaborn scikit-learn matplotlib
```

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KuxC-z4qfmx60Ib1utLHz4V3Xvto0CGe)

## Author

**Fawad Ahmad Bilal**
- GitHub: [@FawadAhmad-bilal](https://github.com/FawadAhmad-bilal)
- Email: fawadahmad19991@gmail.com
