"""
SAP RPT-1 OSS Quick Test Script
=================================

Validates HuggingFace token authentication and runs a quick
classification test using the breast cancer dataset.

Usage:
    # Set your token first
    set HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Run test
    cd code
    python ../scripts/test_sap_rpt1.py

Requirements:
    - Python >= 3.11
    - pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git
    - Hugging Face token with access to SAP/sap-rpt-1-oss

Author: UW MSIM Team
Date: April 2026
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Add code directory to path
sys.path.insert(0, str(project_root / "code"))

# Fix Windows emoji printing issues
if sys.stdout.encoding.lower() != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check all prerequisites before running the test."""
    print("\n" + "=" * 60)
    print("  SAP RPT-1 OSS — Quick Test")
    print("=" * 60)

    # 1. Check Python version
    py_version = sys.version_info
    print(f"\n✅ Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 11):
        print("⚠️  Warning: SAP RPT-1 OSS requires Python >= 3.11")
        print(f"   Your version: {py_version.major}.{py_version.minor}")

    # 2. Check HF token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        print(f"✅ HF Token found: {token[:8]}...{token[-4:]}")
    else:
        print("❌ No HF token found!")
        print("   Set it with: set HUGGING_FACE_HUB_TOKEN=hf_xxx")
        return False

    # 3. Check sap_rpt_oss package
    try:
        import sap_rpt_oss
        print("✅ sap_rpt_oss package installed")
    except ImportError:
        print("❌ sap_rpt_oss not installed!")
        print("   Install with: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git")
        return False

    # 4. Check HF authentication
    try:
        from huggingface_hub import HfApi, login
        login(token=token, add_to_git_credential=False)
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ HF authenticated as: {user_info.get('name', 'unknown')}")
    except Exception as e:
        print(f"❌ HF authentication failed: {e}")
        print("   Make sure you've accepted the license at:")
        print("   https://huggingface.co/SAP/sap-rpt-1-oss")
        return False

    return True


def run_classification_test():
    """Run a classification test on the breast cancer dataset."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sap_rpt_oss import SAP_RPT_OSS_Classifier

    print("\n" + "-" * 60)
    print("  Classification Test: Breast Cancer Dataset")
    print("-" * 60)

    # Load data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\n📊 Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
    print(f"📊 Features: {X.shape[1]}")

    # Initialize model (use small context for quick test)
    print("\n🔧 Initializing SAP RPT-1 OSS Classifier...")
    print("   max_context_size=2048, bagging=1 (fast test mode)")

    start_init = time.time()
    clf = SAP_RPT_OSS_Classifier(max_context_size=2048, bagging=1)
    init_time = time.time() - start_init
    print(f"   Model loaded in {init_time:.2f}s")

    # Fit
    print("\n🏋️ Fitting model (in-context learning)...")
    start_fit = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_fit
    print(f"   Fit completed in {fit_time:.2f}s")

    # Predict
    print("\n🔮 Making predictions...")
    start_pred = time.time()
    predictions = clf.predict(X_test)
    pred_time = time.time() - start_pred
    print(f"   Predictions completed in {pred_time:.2f}s")

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"  Init time: {init_time:.2f}s")
    print(f"  Fit time: {fit_time:.2f}s")
    print(f"  Predict time: {pred_time:.2f}s")
    print(f"  Total time: {init_time + fit_time + pred_time:.2f}s")
    print()
    print(classification_report(y_test, predictions, target_names=['malignant', 'benign']))

    return accuracy


def run_wrapper_test():
    """Run a test using the SAPRPT1HFWrapper from the project."""
    from models.sap_rpt1_hf_wrapper import SAPRPT1HFWrapper
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "-" * 60)
    print("  Wrapper Integration Test: SAPRPT1HFWrapper")
    print("-" * 60)

    # Load data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Use the project wrapper
    wrapper = SAPRPT1HFWrapper(
        task_type='classification',
        max_context_size=2048,
        bagging=1
    )
    wrapper.fit(X_train, y_train)
    predictions = wrapper.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n  ✅ Wrapper test passed! Accuracy: {accuracy:.4f}")
    print(f"  ✅ Fit time: {wrapper.fit_time:.2f}s")

    # Test predict_proba
    try:
        proba = wrapper.predict_proba(X_test)
        print(f"  ✅ predict_proba works! Shape: {proba.shape}")
    except Exception as e:
        print(f"  ⚠️  predict_proba failed: {e}")

    return accuracy


if __name__ == "__main__":
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Fix the issues above and try again.")
        sys.exit(1)

    # Run tests
    try:
        accuracy = run_classification_test()
        wrapper_accuracy = run_wrapper_test()

        print("\n" + "=" * 60)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 60)
        print(f"\n  You can now run experiments with:")
        print(f"    python -m runners.run_experiment --dataset adult --model sap-rpt1-hf")
        print()

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
