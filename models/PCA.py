import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[0] < 2:
            raise ValueError("PCA requires at least two samples")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")

        return X

    def fit(self, X):
        X = self._validate_input(X)

        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)

        if self.n_components is None:
            n_components = max_components
        else:
            if not isinstance(self.n_components, int):
                raise TypeError("n_components must be an integer or None")

            if not 1 <= self.n_components <= max_components:
                raise ValueError(
                    "n_components must satisfy "
                    "1 <= n_components <= min(n_samples, n_features)"
                )

            n_components = self.n_components

        self.n_features_in_ = n_features
        self.n_components_ = n_components
        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        _, singular_values, vt = np.linalg.svd(
            X_centered,
            full_matrices=False
        )

        explained_variance = (
            singular_values ** 2 / (n_samples - 1)
        )

        total_variance = np.sum(explained_variance)

        self.components_ = vt[:n_components]
        self.singular_values_ = singular_values[:n_components]
        self.explained_variance_ = explained_variance[:n_components]

        if total_variance > 0:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(
                n_components
            )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted before transform")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}"
            )

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        if not self.is_fitted_:
            raise RuntimeError(
                "PCA must be fitted before inverse_transform"
            )

        X_transformed = np.asarray(
            X_transformed,
            dtype=float
        )

        if X_transformed.ndim != 2:
            raise ValueError(
                "X_transformed must be a 2D array"
            )

        if X_transformed.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} components, "
                f"but got {X_transformed.shape[1]}"
            )

        return (
            X_transformed @ self.components_
            + self.mean_
        )