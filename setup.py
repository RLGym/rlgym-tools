from setuptools import setup, find_packages

__version__ = None  # This will get replaced when reading version.py
exec(open('rlgym_tools/version.py').read())

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='rlgym_tools',
    packages=find_packages(),
    version=__version__,
    description='Extra tools for RLGym, like SB3 compatibility',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Rolv-Arild Braaten, Lucas Emery and Matthew Allen',
    url='https://rlgym.github.io',
    install_requires=[
        'rlgym>=1.2.0',
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gym', 'reinforcement-learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        "Operating System :: Microsoft :: Windows",
    ],
)
