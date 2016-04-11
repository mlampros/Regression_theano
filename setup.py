from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='Regression_theano',
      version='0.1',
      description='linear and logistic regression in Theano',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
	'Intended Audience :: End Users/Desktop',
	'License :: OSI Approved :: GNU General Public License (GPL)',
	'Natural Language :: English',
	'Operating System :: POSIX :: Linux',
	'Programming Language :: Python :: 2.7',
	'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords='linear logistic regression theano',
      url='https://github.com/mlampros/Regression_Theano',
      author='Mouselimis Lampros',
      packages=['Regression_theano'],
      install_requires=[
          'numpy', 'sklearn', 'scikits.statsmodels', 'theano',
      ],
      include_package_data=True,
      zip_safe=False)
