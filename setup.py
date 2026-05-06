import numpy
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


ext_modules = [
    Extension(
        "connectivity",  # This ensures the module is inside the package
        sources=["connectivity.pyx"],
        include_dirs=[numpy.get_include()],  # Include header files from the package directory
        language="c"
    )
]

setup(
    name="Connectivity",
    version="0.1",
    packages=find_packages(),  # Automatically finds subdirectories with __init__.py
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)