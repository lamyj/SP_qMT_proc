import subprocess

subprocess.check_call([
    "conda", "install", "--yes", "-c", "conda-forge",
    "cmake", "gsl", "cxx-compiler", "make", "nibabel", "numpy", "pkg-config",
    "pybind11", "scipy", "xtensor", "xtensor-blas", "xtensor-python"])
