#!/usr/bin/env python
from distutils.core import setup

setup(
    name="blackbox-backprop",
    version="1.0",
    description="Combinatorial Solvers turned into Torch modules via the method from Differentiation of Blackbox Combinatorial Solvers (Vlastelica et al)",
    author="Michal Rolinek",
    author_email="michalrolinek@gmail.com",
    url=None,
    packages=["blackbox_backprop"],
    install_requires=[
        "torch>=1.0.0",
        "numpy",
        "ray",
        "blossom_v @ git+https://gitlab.tuebingen.mpg.de/mrolinek/blossom_python.git",
    ],
)
