"""
TabPFN Wrapper
==============

Sklearn-compatible wrapper for TabPFN (Tabular Pre-trained Transformers).

TabPFN is a pretrained model for tabular classification using
in-context learning (no training required).

Author: UW MSIM Team
Date: November 2025
"""

import time
import logging
import os
from typing import Optional, Union
import numpy as np
import pandas as pd

# Automatically accept the TabPFN license to prevent browser/socket crashes on Windows
os.environ["TABPFN_LICENSE"] = "accept"
os.environ["TABPFN_ACCEPT_LICENSE"] = "1"

# ── Patch for old TabPFN compatibility with newer torch ──────────────────────
try:
    import torch.nn.modules.transformer
    if not hasattr(torch.nn.modules.transformer, 'Optional'):
        import typing
        torch.nn.modules.transformer.Optional = typing.Optional
        torch.nn.modules.transformer.Any = typing.Any
        torch.nn.modules.transformer.Tuple = typing.Tuple
        torch.nn.modules.transformer.List = typing.List
except (ImportError, AttributeError):
    pass

# ── Patch for old TabPFN compatibility with newer sklearn ────────────────────
try:
    import sklearn.utils.validation
    def _patch_validation(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'force_all_finite' in kwargs:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return func(*args, **kwargs)
        return wrapper
    sklearn.utils.validation.check_X_y = _patch_validation(sklearn.utils.validation.check_X_y)
    sklearn.utils.validation.check_array = _patch_validation(sklearn.utils.validation.check_array)
except (ImportError, AttributeError):
    pass

from .base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class TabPFNWrapper(BaseModelWrapper):
    """
    TabPFN (Tabular Prior-Fitted Networks) wrapper.

    TabPFN uses pretrained transformers for zero-shot tabular prediction.
    Works best on datasets with <1000 samples and <100 features.

    Parameters
    ----------
    task_type : str, default='classification'
        Task type (only 'classification' supported by TabPFN)
    n_ensemble : int, default=1
        Number of ensemble members
    device : str, default='auto'
        Device: 'cpu', 'cuda', or 'auto'
    random_state : int, default=42
        Random seed
    """

    def __init__(
        self,
        task_type: str = 'classification',
        n_ensemble: int = 1,
        device: str = 'auto',
        random_state: int = 42
    ):
        super().__init__(task_type=task_type, random_state=random_state)

        if task_type != 'classification':
            raise ValueError("TabPFN only supports classification tasks")

        self.n_ensemble = n_ensemble
        self.device = device

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'TabPFNWrapper':
        """
        Fit TabPFN (stores training data for in-context learning).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training features (max 1000 samples, 100 features)
        y : pd.Series or np.ndarray, shape (n_samples,)
            Training target

        Returns
        -------
        self : TabPFNWrapper
            Fitted model
        """
        self._validate_input(X, y)

        # Check TabPFN constraints
        if X.shape[0] > 1024:
            logger.warning(f"TabPFN strictly requires <= 1024 samples to avoid Memory OOM. Subsampling {X.shape[0]} to 1024 samples.")
            sample_idx = np.random.RandomState(self.random_state).choice(
                len(X), 1024, replace=False
            )
            if isinstance(X, pd.DataFrame):
                X = X.iloc[sample_idx]
            else:
                X = X[sample_idx]
            
            if isinstance(y, pd.Series):
                y = y.iloc[sample_idx]
            else:
                y = y[sample_idx]

        if X.shape[1] > 100:
            logger.warning(f"TabPFN strictly requires <= 100 features. Truncating {X.shape[1]} to 100 features.")
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, :100]
            else:
                X = X[:, :100]
            self.truncated_features_ = True
        else:
            self.truncated_features_ = False

        logger.info(f"Fitting TabPFN on {X.shape[0]} samples...")
        start_time = time.time()

        try:
            from tabpfn import TabPFNClassifier

            import torch
            import tabpfn
            
            actual_device = 'cuda' if (self.device == 'auto' and torch.cuda.is_available()) else ('cpu' if self.device == 'auto' else self.device)
            
            if hasattr(tabpfn, '__version__') and tabpfn.__version__.startswith('0.1'):
                self.model = TabPFNClassifier(device=actual_device, N_ensemble_configurations=self.n_ensemble)
            else:
                self.model = TabPFNClassifier(device=actual_device)

            # Fit model
            self.model.fit(X, y)

            self.is_fitted = True
            self.fit_time = time.time() - start_time

            logger.info(f"TabPFN fitted in {self.fit_time:.2f} seconds")

        except ImportError:
            logger.error("TabPFN not installed")
            raise ImportError("Install TabPFN with: pip install tabpfn")
        except Exception as e:
            logger.error(f"Error fitting TabPFN: {e}")
            raise

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with TabPFN.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Test features

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_input(X)

        if getattr(self, 'truncated_features_', False) and X.shape[1] > 100:
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, :100]
            else:
                X = X[:, :100]

        logger.info(f"Predicting on {X.shape[0]} samples with TabPFN...")
        start_time = time.time()

        try:
            predictions = self.model.predict(X)
            self.predict_time = time.time() - start_time

            logger.info(f"Predictions complete in {self.predict_time:.2f} seconds")

            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def _predict_proba_impl(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities with TabPFN.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Test features

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Class probabilities
        """
        if getattr(self, 'truncated_features_', False) and X.shape[1] > 100:
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, :100]
            else:
                X = X[:, :100]

        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({
            'n_ensemble': self.n_ensemble,
            'device': self.device
        })
        return params
