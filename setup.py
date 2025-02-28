from setuptools import setup, find_packages

setup(
    name="humanitarian-analysis",
    version="0.1.0",
    description="Library for humanitarian situation analysis using NLP",
    author="Reporter.ai (https://reporterai.org)",
    author_email="reporter.ai@boldcode.io",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "datasets",
        "pandas",
        "tqdm",
        "numpy",
        # Git dependencies removed from here
    ],
    python_requires=">=3.7",
    license="AGPL-3.0",
)