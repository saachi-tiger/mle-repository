from setuptools import setup, find_packages

setup(
    name='Housing Package',
    version='0.1.0',
    author='Saachi Mohanty',
    author_email='saachi.mohanty@tigeranalytics.com',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/saachi-tiger/mle-repository',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ],
)
