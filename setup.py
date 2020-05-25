import setuptools

setuptools.setup(name='tanami',
                 version='0.1',
                 description='Miscellaneous utilities to manipulate data',
                 url='http://github.com/akiatoji/tanami',
                 author='Aki Atoji',
                 author_email='aki@zittatech.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=['pandas>=1.0', 'numpy>=1.18', 'scikit-learn>=0.22' ],
                 python_requires='>=3.6',
                 zip_safe=False)
