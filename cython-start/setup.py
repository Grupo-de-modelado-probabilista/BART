from distutils.core import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace

setup(
    name="new_idx_data_points_cython",
    ext_modules=cythonize(
        "new_idx_data_points_cython.pyx", language_level=3, annotate=True
    ),
    # accepts a glob pattern
)
