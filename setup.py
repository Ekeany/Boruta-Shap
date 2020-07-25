from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="BorutaShap",
    version="1.0.12",
    description="A feature selection algorithm.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ekeany/Boruta-Shap",
    author="Eoghan Keany",
    author_email="egnkeany@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    py_modules = ["BorutaShap"],
    package_dir = {"" : "src"},
    install_requires=["scikit-learn","tqdm",
                      "statsmodels","matplotlib",
                      "pandas","numpy","shap<=0.34.0,>=0.32.0","seaborn",
                      "scipy"],
)
