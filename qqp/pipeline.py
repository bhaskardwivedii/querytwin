"""Shared glue between training and inference: how each model's input matrix is built."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def make_inputs(kind: str, X_sparse, X_dense, scaler, xgb_features: str):
    """Build the design matrix for a model family.

    lr  : sparse TF-IDF pair vectors + standardised dense features
    rf  : dense engineered features only (trees on 100k+ sparse columns are impractical)
    xgb : dense features (+ sparse TF-IDF pair vectors when xgb_features == 'dense+tfidf')
    """
    if kind == "lr":
        return sp.hstack([X_sparse, sp.csr_matrix(scaler.transform(X_dense))],
                         format="csr", dtype=np.float32)
    if kind == "rf":
        return X_dense
    if kind == "xgb":
        if xgb_features == "dense":
            return X_dense
        return sp.hstack([sp.csr_matrix(X_dense), X_sparse], format="csr", dtype=np.float32)
    raise ValueError(kind)
