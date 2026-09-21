# Machine Learning from Scratch

A small educational machine-learning library implemented with NumPy. The project exposes the optimization steps behind classical models instead of wrapping ready-made estimators. Scikit-learn is used only in reproducible comparison scripts.

## Implemented models

| Area | Model | Implementation details |
| --- | --- | --- |
| Regression | Linear regression | Gradient descent and stable least-squares solution |
| Classification | Logistic regression | Stable sigmoid/log-loss, L1 proximal step, L2 regularization |
| Clustering | K-Means | K-Means++, empty-cluster recovery, `n_init`, deterministic seeds |
| Probabilistic models | Gaussian mixture | Full-covariance EM, log-domain responsibilities, AIC/BIC, sampling |
| Dimensionality reduction | PCA | SVD, transform and inverse transform, variance ratios |
| Kernel methods | SVM | Linear SGD and kernel SMO optimizers |
| Tree models | CART | Classification/regression, sample weights, validation-set pruning |
| Ensembles | AdaBoost | Binary boosting with weighted CART stumps |
| Neural networks | MLP classifier | One hidden layer and manual backpropagation |

The implementations favor clarity and verification over production performance. Logistic regression, SVM, AdaBoost, and the MLP currently support binary classification. GMM currently uses full covariance matrices.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from models import LinearRegression

rng = np.random.RandomState(42)
X = rng.normal(size=(200, 2))
y = 1.5 + X @ np.array([2.0, -3.0]) + rng.normal(scale=0.1, size=200)

model = LinearRegression(method="closed_form").fit(X, y)
print(model.coef_, model.intercept_)
print(model.predict(X[:3]))
```

Run the smoke demo:

```bash
python main.py
```

Run all correctness tests:

```bash
pytest
```

Run the deterministic sklearn comparison:

```bash
python benchmarks/compare_sklearn.py
```

The benchmark writes machine-readable results to `benchmarks/results.json`. Runtime measurements depend on the machine and should be treated as descriptive rather than universal.

Reference results on the fixed synthetic datasets:

| Model | This project | scikit-learn |
| --- | ---: | ---: |
| Linear regression (MSE, lower is better) | 0.036414 | 0.036414 |
| Logistic regression (accuracy) | 0.8433 | 0.8400 |
| K-Means (ARI) | 1.0000 | 1.0000 |
| GMM (average log likelihood) | -3.752615 | -3.752615 |
| PCA (explained-variance ratio sum) | 0.486454 | 0.486454 |
| CART (accuracy) | 0.6633 | 0.6567 |

These numbers are regression checks on small synthetic data, not evidence that the implementations generally outperform scikit-learn. The benchmark also exposes a useful engineering gap: this CART implementation is much slower because it enumerates split candidates in Python.

## Verification

The test suite covers:

- finite-difference gradient checking for logistic regression;
- singular-design linear least squares;
- deterministic K-Means initialization and non-increasing inertia;
- normalized GMM responsibilities and non-decreasing EM likelihood;
- PCA orthogonality and exact full-rank reconstruction;
- CART classification, regression pruning, and sample weights;
- AdaBoost training behavior;
- consistent SVM `decision_function`/`predict` preprocessing;
- a small MLP learning test;
- metric edge cases.

GitHub Actions runs the tests and sklearn benchmark on Python 3.10, 3.11, and 3.12.

## Repository structure

```text
.
├── models/                  # NumPy model implementations
│   ├── linear_model.py
│   ├── kmeans.py
│   ├── gmm.py
│   ├── PCA.py
│   ├── neural_network.py
│   ├── SVM_model/
│   └── tree/
├── data/synthetic/          # Reproducible toy-data generators
├── experiments/             # Model-specific exploratory scripts
├── benchmarks/              # Deterministic sklearn comparisons
├── tests/                   # Automated correctness tests
├── .github/workflows/       # CI configuration
├── main.py                  # Minimal runnable example
└── pyproject.toml           # Package and dependency metadata
```

## Design choices

- Estimators follow a consistent `fit(...)->self` and `predict(...)` convention.
- Learned public attributes use trailing underscores where practical (`coef_`, `labels_`, `weights_`).
- Input dimensions, labels, finite values, and fitted state are checked at API boundaries.
- Randomized models accept `random_state`; benchmarks use fixed seeds.
- Probability calculations in logistic regression and GMM use numerically stable formulations.

Some legacy aliases remain (`linearregression`, `Kmeans`, `BPNet`) so older experiment code continues to run.

## Current limitations and roadmap

This remains an educational project rather than a production library. The next useful extension is a focused computer-vision experiment rather than another list of unrelated algorithms:

1. evaluate PCA/HOG features with linear and RBF SVM on Fashion-MNIST;
2. add a small NumPy Conv2D implementation with numerical gradient checks;
3. compare MLP and CNN accuracy, runtime, initialization, and failure cases;
4. publish plots and a concise experiment report.

## Author

Hu Kai — [GitHub @hukaihukai123](https://github.com/hukaihukai123)
