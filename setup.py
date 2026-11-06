"""Setup script for vectorphile."""
from setuptools import setup, find_packages

setup(
    name='vectorphile',
    version='0.1.0',
    description='Unified ML/DL Python Library — replaces PyTorch + scikit-learn + NumPy + Pandas + SciPy',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='vectorphile',
    author_email='saadsalmanakram1@gmail.com',
    url='https://github.com/vectorphile/vectorphile',
    license='Apache-2.0',
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*', 'csrc*']),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
    ],
    extras_require={
        'full': [
            'scipy>=1.10.0',
            'pyarrow>=12.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machine learning deep learning neural network numpy tensor dataframe',
    project_urls={
        'Bug Reports': 'https://github.com/vectorphile/vectorphile/issues',
        'Source': 'https://github.com/vectorphile/vectorphile',
    },
)
