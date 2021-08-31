import codecs
import os

import setuptools


def read(file_path):
    here = os.path.abspath(os.path.dirname(__file__))

    with codecs.open(os.path.join(here, file_path), 'r') as fp:
        return fp.read()


def get_version(file_path):
    for line in read(file_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


install_requires = ['numpy >= 1.19.0',
                    'setuptools >= 49.0.0',
                    'scikit-learn >= 0.23.0',
                    'scipy >= 1.5.0',
                    'joblib >= 0.16.0',
                    ]

docs_require = ['sphinx >= 1.4',
                'sphinx_rtd_theme']

setuptools.setup(name='diffprivlib',
                 version=get_version("diffprivlib/__init__.py"),
                 description='IBM Differential Privacy Library',
                 long_description=read("README.md"),
                 long_description_content_type='text/markdown',
                 author='Naoise Holohan',
                 author_email='naoise.holohan@ibm.com',
                 url='https://github.com/IBM/differential-privacy-library',
                 license='MIT',
                 install_requires=install_requires,
                 extras_require={
                     'docs': docs_require
                 },
                 python_requires='>=3.7',
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Education',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved',
                     'License :: OSI Approved :: MIT License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3 :: Only',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Topic :: Software Development :: Libraries',
                     'Topic :: Software Development :: Libraries :: Python Modules',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Security',
                 ],
                 packages=setuptools.find_packages())
