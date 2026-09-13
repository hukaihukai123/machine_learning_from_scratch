# Machine Learning from Scratch

> A small educational machine-learning library implemented primarily with NumPy.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243.svg)](https://numpy.org/)

This repository is my first attempt to turn the mathematical ideas behind classical machine-learning algorithms into working code. Instead of calling ready-made estimators, the core training procedures—including gradient descent, EM, CART splitting, backpropagation, SGD, and SMO—are implemented manually.

The project is intended for learning and experimentation rather than production use. Scikit-learn is used only for dataset generation, metric comparison, and reference baselines in several demos.

## Features

| Category | Implementation | Highlights |
| --- | --- | --- |
| Regression | Linear Regression | Gradient descent and closed-form solution; optional intercept |
| Classification | Logistic Regression | Binary classification; L1/L2 regularization options |
| Clustering | K-Means | Random centroid initialization, empty-cluster handling, convergence tracking |
| Probabilistic Models | Gaussian Mixture Model | K-Means initialization, EM algorithm, posterior probabilities, sampling |
| Dimensionality Reduction / Classification | Linear Discriminant Analysis | Binary LDA with a regularized within-class scatter matrix |
| Kernel Methods | Support Vector Machine | Linear, RBF, polynomial, sigmoid, and custom kernels; SGD and SMO optimizers |
| Tree Models | CART Decision Tree | Classification and regression; pre-pruning and validation-set post-pruning |
| Neural Networks | BP Neural Network | One-hidden-layer binary classifier with ReLU, sigmoid, and manual backpropagation |
| Utilities | Metrics and synthetic data | Regression/classification metrics, reports, and reproducible data generators |

## Repository Structure

```text
machine_learning_from_scratch/
├── config.py                      # Global random-state configuration
├── data/
│   ├── real/                      # Placeholder for real-world datasets
│   └── synthetic/                 # Synthetic regression and clustering data
├── experiments/
│   ├── 01_linear_regression.py
│   ├── 02_logistic_regression.py
│   ├── 03_gmm.py
│   ├── 04_svm.py
│   └── 05_tree_ensemble.py
├── models/
│   ├── linear_model.py            # Linear and logistic regression
│   ├── kmeans.py                  # K-Means
│   ├── gmm.py                     # Gaussian mixture model
│   ├── LDA.py                     # Linear discriminant analysis
│   ├── SVM_model/
│   │   ├── core/                  # SVM model and kernels
│   │   └── optimizer/             # SGD and SMO optimizers
│   ├── tree/decisiontreeCART.py   # CART decision tree
│   └── neural network/BP.py       # Backpropagation neural network
├── notebooks/playground.ipynb
├── utils/metrics.py
└── requirements.txt
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hukaihukai123/machine_learning_from_scratch.git
cd machine_learning_from_scratch
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Then install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run commands from the repository root. Because the experiment scripts import top-level project packages, add the repository root to `PYTHONPATH` when running a script directly.

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
python experiments/03_gmm.py
python experiments/04_svm.py
python experiments/05_tree_ensemble.py
```

```bash
# Linux / macOS
PYTHONPATH=. python experiments/03_gmm.py
PYTHONPATH=. python experiments/04_svm.py
PYTHONPATH=. python experiments/05_tree_ensemble.py
```

The standalone LDA and BP demos can also be run directly:

```bash
python models/LDA.py
python "models/neural network/BP.py"
```

## Usage Examples

### Linear Regression

```python
from data.synthetic.linear import make_linear_data
from models.linear_model import linearregression
from utils.metrics import get_all_metrics_regression

X, y, theta_true = make_linear_data(
    n_samples=500,
    n_features=2,
    noise=0.5,
    random_state=42,
)

model = linearregression(
    method="gd",
    learning_rate=1e-3,
    n_iterations=2000,
)
model.fit(X, y)
y_pred, theta_pred = model.predict(X)

print("True parameters:", theta_true)
print("Learned parameters:", theta_pred)
print(get_all_metrics_regression(y, y_pred))
```

Set `method="closed_form"` (or any value other than `"gd"` in the current implementation) to use the normal-equation solution.

### Gaussian Mixture Model

```python
from data.synthetic.gmm_data import generate_gmm_data
from models.gmm import GaussianMixtureModel

X, y_true = generate_gmm_data(
    n_samples=600,
    weights=[0.4, 0.6],
    means=[[0, 0], [4, 4]],
    covariances=[[[1, 0], [0, 1]], [[1, 0.3], [0.3, 1]]],
    random_state=42,
)

model = GaussianMixtureModel(n_components=2, max_iter=100)
model.fit(X)

labels = model.predict(X)
probabilities = model.predict_proba(X)
new_samples = model.sample(10)
```

### Kernel SVM with SMO

```python
from models.SVM_model.core.kernel import Kernel
from models.SVM_model.core.svm import SVM
from models.SVM_model.optimizer.smo import SMOOptimizer

kernel = Kernel(kernel_type="rbf", gamma=0.5)
model = SVM(C=1.0, kernel=kernel, optimizer=SMOOptimizer())
model.fit(X_train, y_train)  # labels: {-1, 1} or {0, 1}
y_pred = model.predict(X_test)
```

For a linear SVM trained with stochastic gradient descent:

```python
from models.SVM_model.optimizer.sgd import SGDOptimizer

model = SVM(
    C=1.0,
    kernel=None,
    optimizer=SGDOptimizer(lr=0.01, epochs=100),
)
```

### CART Decision Tree

```python
from models.tree.decisiontreeCART import DecisionTreeCART

tree = DecisionTreeCART(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    min_impurity_decrease=1e-3,
    discrete=True,
)

# X_val and y_val are optional and enable post-pruning.
tree.fit(X_train, y_train, X_val, y_val)
y_pred = tree.predict(X_test)
```

Use `discrete=False` to build a regression tree.

## Implementation Notes

- Model optimization is written manually with NumPy; the repository does not wrap scikit-learn estimators as its own models.
- The SVM separates the model, kernel, and optimizer into independent components, making SGD and SMO interchangeable.
- GMM training uses K-Means initialization before alternating between the E-step and M-step.
- CART supports both early-stopping constraints and reduced-error post-pruning with a validation set.
- Training histories such as `loss_history`, `lost_history`, and `log_likelihood_history` are retained for inspection.
- Randomness can be controlled globally through `config.py` or locally through supported `random_state` parameters.

## Current Scope

This is an evolving learning project. The current focus is on readable implementations that expose the underlying algorithmic steps. Possible future improvements include:

- consistent scikit-learn-style APIs and naming;
- automated unit tests and benchmark reports;
- vectorized and numerically stable GMM calculations using log-sum-exp;
- multiclass extensions for logistic regression, LDA, and SVM;
- additional preprocessing, validation, and visualization utilities;
- packaging and continuous integration.

## Author

**Hu Kai**

- GitHub: [@hukaihukai123](https://github.com/hukaihukai123)

Contributions, suggestions, and issue reports are welcome.
