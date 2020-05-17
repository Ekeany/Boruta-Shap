from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="BorutaShap",
    version="1.0.0",
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
    packages=["BorutaShap.py"],
    include_package_data=True,
    install_requires=["sklearn","tqdm",
                      "statsmodels","matplotlib",
                      "pandas","numpy","shap","seaborn",
                      "scipy"],
)
