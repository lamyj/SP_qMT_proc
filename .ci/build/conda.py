import subprocess

subprocess.check_call([
    "conda", "install", "--yes", "-c", "conda-forge",
    "cmake", "gsl", "gxx", "make", "nibabel", "numpy", "pybind11",
    "scipy", "xtensor", "xtensor-blas", "xtensor-python"])
