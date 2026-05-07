from setuptools import setup, find_packages

setup(
    name="sap-rpt1",
    version="0.1.0",
    package_dir={"": "code"},
    packages=find_packages(where="code"),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "scipy>=1.14.1",
        "matplotlib>=3.9.2",
        "seaborn>=0.13.2",
        "pyyaml>=6.0.2",
        "openml>=0.14.2",
        "tqdm>=4.67.1",
        "joblib>=1.4.2",
        "psutil>=6.1.1",
    ],
    extras_require={
        "models": [
            "torch>=2.7.0",
            "transformers>=4.52.4",
            "accelerate>=1.6.0",
            "huggingface-hub>=0.30.2",
            "datasets>=3.5.0",
            "pyarrow>=20.0.0",
            "torcheval>=0.0.7",
            "python-dotenv>=1.0.1",
            "sap-rpt-oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@v1.1.2",
        ],
        "baselines": [
            "xgboost>=2.0.3",
            "catboost>=1.2.3",
            "lightgbm>=4.3.0",
            "autogluon.tabular[all]>=1.0.0",
            "tabpfn>=0.1.9",
        ],
    },
    python_requires=">=3.11",
)
