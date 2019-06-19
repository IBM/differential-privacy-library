import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy >= 1.15.0',
                    'setuptools >= 39.0.1',
                    'scikit-learn >= 0.20.3',
                    'scipy >= 1.2.1',
                    'joblib >= 0.13.2',
                    ]

docs_require = ['sphinx >= 1.4',
                'sphinx_rtd_theme']

setuptools.setup(name='diffprivlib',
                 version='0.1.1',
                 description='IBM Differential Privacy Library',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 author='Naoise Holohan',
                 author_email='naoise.holohan@ibm.com',
                 url='https://github.com/IBM/differential-privacy-library',
                 license='MIT',
                 install_requires=install_requires,
                 extras_require={
                     'docs': docs_require
                 },
                 classifiers=[
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Education',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3 :: Only',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Topic :: Software Development :: Libraries',
                     'Topic :: Software Development :: Libraries :: Python Modules',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Security',
                 ],
                 packages=setuptools.find_packages())
