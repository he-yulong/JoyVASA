from setuptools import setup, find_packages

setup(
    name='joyvasa',
    version='0.1',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        # Optional: add required packages here
    ],
    author='Yulong He',
    description='JoyVASA project package',
    python_requires='>=3.7',
)
