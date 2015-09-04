from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='Mu2E',
    version='0.1',
    description='Mu2E Analysis Software',
    url='https://github.com/brovercleveland/Mu2E',
    author='Brian Pollack',
    author_email='brianleepollack@gmail.com',
    license='NU',
    packages=find_packages(),
    install_requires=[
      'numpy',
      'scipy',
      'pandas',
      'lmfit'],
    include_package_data=True,
    zip_safe=False)
