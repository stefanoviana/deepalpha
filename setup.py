from setuptools import setup, find_packages

setup(
    name="deepalpha",
    version="11.3.0",
    description="AI crypto trading bot for Bybit, Binance, OKX, Gate.io, KuCoin, Bitget — 70.9% accuracy, 72 ML features, pump scanner",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="DeepAlpha",
    author_email="support@deepalphabot.com",
    url="https://deepalphabot.com",
    project_urls={
        "Cloud Dashboard": "https://deepalphabot.com/cloud",
        "GitHub": "https://github.com/stefanoviana/deepalpha",
        "Discord": "https://discord.gg/P4yX686m",
    },
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="crypto trading bot ai machine-learning bybit binance xgboost lightgbm",
)
