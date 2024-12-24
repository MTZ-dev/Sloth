from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="slothpy._general_utilities._lapack",
        sources=["slothpy/_general_utilities/_lapack.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-ffast-math", "-funroll-loops", "-flto"],
        extra_link_args=["-fopenmp", "-flto"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), ('CYTHON_CCOMPLEX', '0')],
    )
]

setup(
    name="slothpy",
    version="0.3.0",
    packages=find_packages(include=["slothpy", "slothpy.*"]),
    setup_requires=["cython>=3.0.10"],
    ext_modules=cythonize(extensions, compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "language_level": "3",
    }
        ),
    zip_safe=False,
    include_package_data=True,
    package_data={
        'slothpy': ['static/data', 'static/slothpy_3.ico', 'settings.ini'],
    },
)
