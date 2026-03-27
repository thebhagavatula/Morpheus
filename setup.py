from setuptools import setup, find_packages

setup(
    name="cross-sectional-expected-return-model",
    version="1.0.0",
    description="Cross-Sectional Expected Return Forecasting Pipeline with Fama-MacBeth Factor Premia",
    author="Karthik",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "yfinance>=0.2.20",
        "streamlit>=1.22.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "simfin": ["simfin>=0.10.2"],
        "numba": ["numba>=0.57.0"],
    },
)
