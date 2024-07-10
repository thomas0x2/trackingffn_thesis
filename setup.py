from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

lib_include = os.path.abspath("lib")

ext_modules = [
    Pybind11Extension(
        "neural_network",
        ["neural_network.cpp"],
        include_dirs=[
            pybind11.get_include(),
            lib_include,
        ],
        extra_compile_args=['-std=c++11', '-stdlib=libc++']# Add this line
    ),
]

setup(
    name="neural_network",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
