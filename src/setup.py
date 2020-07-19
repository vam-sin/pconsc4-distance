from __future__ import division
import os
from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize
import numpy as np

# Check wheter we have pyx or c files:
base_path = os.path.dirname(os.path.abspath(__file__))

# A sdist will have C files, use those:
if os.path.exists(os.path.join(base_path, 'parsing/_load_data.c')):
    use_cython = False
else:
    # It appears we are on git, go ahead and cythonice everything
    use_cython = True

flags = "-O2 -march=native -pipe -mtune=native".split()

if use_cython:
    extensions = [
        Extension(
            'parsing._load_data', ['parsing/_load_data.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=flags,
            extra_link_args=flags),
        Extension(
            'parsing._mi_info', ['parsing/_mi_info.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=flags,
            extra_link_args=flags)
    ]
else:
    extensions = [
        Extension(
            'parsing._load_data', ['parsing/_load_data.c'],
            include_dirs=[np.get_include()],
            extra_compile_args=flags,
            extra_link_args=flags),
        Extension(
            'parsing._mi_info', ['parsing/_mi_info.c'],
            include_dirs=[np.get_include()],
            extra_compile_args=flags,
            extra_link_args=flags)
    ]

setup(
    name='pconsc4',
    version='0.4',
    description='',
    url='https://github.com/ElofssonLab/PconsC4',
    author='Mirco Michel and David Menéndez Hurtado',
    author_email='davidmenhur@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    include_dirs=[np.get_include()],
    include_package_data=True,
    ext_modules=cythonize(extensions),
    setup_requires=['numpy', 'Cython'],
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: '
        'GNU General Public License v3 (GPLv3)'
    ],
    zip_safe=False)

