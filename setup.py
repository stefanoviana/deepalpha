from setuptools import setup, find_packages

setup(
    name="deepalpha",
    version="11.0.0",
    description="AI-Powered Crypto Trading Bot for Bybit",
    author="DeepAlpha",
    url="https://deepalphabot.com",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "lightgbm>=4.0.0",
        "scikit-learn>=1.3.0",
        "ccxt>=4.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "xgboost>=2.0.0",
    ],
)
