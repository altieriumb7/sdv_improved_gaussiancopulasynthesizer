from setuptools import setup, find_packages

setup(
    name="new_sdv",
    version="0.1.0",
    description="Generazione di dati sintetici con Gaussian Copula",
    author="Nome Autore",
    author_email="email@example.com",
    url="https://github.com/utente/new_sdv",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "torch>=1.8.0",
        "scipy>=1.5.0",
        "plotly>=5.23"
    ],
    python_requires=">=3.7",
)
