from setuptools import setup, find_packages

setup(
    name='keras-position-wise-feed-forward',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-position-wise-feed-forward',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Feed forward layer implemented in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)