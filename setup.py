from setuptools import find_packages
from setuptools import setup


setup(
    name="wdm3d",
    version="0.1",
    author="qgq",
    packages=find_packages(exclude=("test",)),
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

