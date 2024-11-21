import os

from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='dtcwt-yx', # dtcwt has been used
    version='1.0.0',
    author="YaoxinHUANG",
    author_email="simon.niedermayr@tum.de",
    description=("Discrete Wavelet Transform for Pytorch"),
    license="MIT",
    keywords="pytorch, DWT, IDWT, wavelet, 3d, dtcwt, 3D DTCWT",
    url="https://github.com/YaoxinHUANG/3d-dtcwt",
    packages=find_packages(),
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
    include_package_data=True,
    install_requires=['torch','PyWavelets','numpy'],
    tests_require=['pytest','numpy','PyWavelets','torch'],
)
