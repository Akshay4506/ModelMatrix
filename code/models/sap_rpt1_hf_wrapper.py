"""
SAP RPT-1 OSS Wrapper (Hugging Face Authenticated)
====================================================

Sklearn-compatible wrapper for SAP RPT-1-OSS via Hugging Face.

This wrapper uses the official `sap_rpt_oss` package with HF token
authentication for downloading gated model weights.

SAP RPT-1 OSS is a tabular in-context learning model — it does NOT
use text generation. It accepts DataFrames/arrays and produces
predictions directly on structured tabular data.

Requirements:
    - Python >= 3.11
    - pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git
    - Hugging Face token with access to SAP/sap-rpt-1-oss

Author: UW MSIM Team
Date: April 2026
"""

import os
import time
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


def _authenticate_huggingface(token: Optional[str] = None) -> str:
    """
    Authenticate with Hugging Face Hub using token.

    Token resolution order:
        1. Explicit `token` parameter
        2. HUGGING_FACE_HUB_TOKEN environment variable
        3. HF_TOKEN environment variable
        4. Previously saved token via `huggingface-cli login`

    Parameters
    ----------
    token : str, optional
        Explicit HF token to use

    Returns
    -------
    str
        The resolved token

    Raises
    ------
    RuntimeError
        If no valid token is found
    """
    from huggingface_hub import login, HfApi

    # Resolve token from multiple sources
    resolved_token = (
        token
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
    )

    if resolved_token:
        try:
            login(token=resolved_token, add_to_git_credential=False)
            logger.info("✅ Hugging Face authentication successful (via token)")
            return resolved_token
        except Exception as e:
            raise RuntimeError(
                f"Hugging Face authentication failed: {e}\n"
                "Ensure your token is valid and you have accepted the license at:\n"
                "  https://huggingface.co/SAP/sap-rpt-1-oss"
            )

    # Check if already logged in via CLI
    try:
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"✅ Hugging Face authenticated as: {user_info.get('name', 'unknown')}")
        return ""  # Already authenticated
    except Exception:
        pass

    raise RuntimeError(
        "No Hugging Face token found. Please set one of:\n"
        "  1. Environment variable: set HUGGING_FACE_HUB_TOKEN=hf_xxx\n"
        "  2. Environment variable: set HF_TOKEN=hf_xxx\n"
        "  3. Run: huggingface-cli login\n\n"
        "You must also accept the model license at:\n"
        "  https://huggingface.co/SAP/sap-rpt-1-oss"
    )


class SAPRPT1HFWrapper(BaseModelWrapper):
    """
    SAP RPT-1 OSS (Hugging Face) wrapper for tabular prediction.

    Uses the official `sap_rpt_oss` package with in-context learning.
    The model automatically handles:
        - Column/cell embeddings via built-in LLM
        - Missing values
        - CPU/GPU auto-detection (GPU not required)

    Parameters
    ----------
    task_type : str, default='classification'
        Task type: 'classification' or 'regression'
    max_context_size : int, default=4096
        Maximum number of context rows for in-context learning.
        Higher = better accuracy but more memory/time.
        Recommended: 2048 (light), 4096 (balanced), 8192 (best)
    bagging : int or 'auto', default=4
        Number of bagging iterations for prediction stability.
        Use 1 for fast inference, 4-8 for best accuracy.
        'auto' = automatically determined based on dataset size.
    hf_token : str, optional
        Explicit Hugging Face token. If not provided, reads from
        HUGGING_FACE_HUB_TOKEN or HF_TOKEN environment variable.
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(
        self,
        task_type: str = 'classification',
        max_context_size: int = 4096,
        bagging: Union[int, str] = 4,
        hf_token: Optional[str] = None,
        random_state: int = 42
    ):
        super().__init__(task_type=task_type, random_state=random_state)
        self.max_context_size = max_context_size
        self.bagging = bagging
        self.hf_token = hf_token

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'SAPRPT1HFWrapper':
        """
        Fit SAP RPT-1 OSS model.

        Note: SAP RPT-1 uses in-context learning, so "fitting" stores
        the training data for retrieval during inference. The model
        weights are pretrained and NOT updated.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training features
        y : pd.Series or np.ndarray, shape (n_samples,)
            Training target

        Returns
        -------
        self : SAPRPT1HFWrapper
            Fitted model
        """
        self._validate_input(X, y)

        logger.info(
            f"Fitting SAP RPT-1 OSS on {X.shape[0]} samples, "
            f"{X.shape[1]} features (max_context={self.max_context_size}, "
            f"bagging={self.bagging})..."
        )
        start_time = time.time()

        try:
            # Authenticate with Hugging Face (downloads gated model weights)
            _authenticate_huggingface(self.hf_token)

            # Import here to avoid import errors in environments without sap_rpt_oss
            from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

            # Initialize appropriate model based on task type
            if self.task_type == 'classification':
                self.model = SAP_RPT_OSS_Classifier(
                    max_context_size=self.max_context_size,
                    bagging=self.bagging
                )
            else:
                self.model = SAP_RPT_OSS_Regressor(
                    max_context_size=self.max_context_size,
                    bagging=self.bagging
                )

            # Fit model (stores training data for in-context learning)
            self.model.fit(X, y)

            self.is_fitted = True
            self.fit_time = time.time() - start_time

            logger.info(f"✅ SAP RPT-1 OSS fitted in {self.fit_time:.2f} seconds")

        except ImportError as e:
            logger.error(f"SAP RPT-1 OSS package not installed: {e}")
            raise ImportError(
                "sap-rpt-1-oss not found. Install with:\n"
                "  pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git\n\n"
                "Requires Python >= 3.11"
            )
        except Exception as e:
            logger.error(f"Error fitting SAP RPT-1 OSS: {e}")
            raise

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions with SAP RPT-1 OSS.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Test features

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values or class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_input(X)

        logger.info(f"Predicting on {X.shape[0]} samples with SAP RPT-1 OSS...")
        start_time = time.time()

        try:
            predictions = self.model.predict(X)

            # Convert list to numpy array if needed
            if isinstance(predictions, list):
                predictions = np.array(predictions)

            self.predict_time = time.time() - start_time
            logger.info(f"✅ Predictions complete in {self.predict_time:.2f} seconds")

            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def _predict_proba_impl(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities with SAP RPT-1 OSS.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Test features

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")

        try:
            proba = self.model.predict_proba(X)

            # Convert to numpy if needed
            if not isinstance(proba, np.ndarray):
                proba = np.array(proba)

            return proba

        except AttributeError:
            # Fallback: one-hot encode predictions if predict_proba unavailable
            logger.warning(
                "predict_proba not available, using one-hot encoding of predictions"
            )
            predictions = self.model.predict(X)
            if isinstance(predictions, list):
                predictions = np.array(predictions)

            classes = np.unique(predictions)
            n_samples = len(predictions)
            n_classes = len(classes)
            proba = np.zeros((n_samples, n_classes))

            class_to_idx = {c: i for i, c in enumerate(classes)}
            for i, pred in enumerate(predictions):
                proba[i, class_to_idx[pred]] = 1.0

            return proba

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility)."""
        params = super().get_params(deep)
        params.update({
            'max_context_size': self.max_context_size,
            'bagging': self.bagging,
            'hf_token': self.hf_token
        })
        return params
