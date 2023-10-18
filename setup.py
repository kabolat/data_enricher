from setuptools import setup

setup(
    name='data_enricher',
    version='0.1.0',
    packages=['energy_data_enricher'],
    url='https://github.com/kabolat/energy_data_enricher',
    author='Kutay Bolat',
    author_email='k.bolat@tudelft.nl',
    description='Probabilistic modelling tools for energy data enrichment',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=['torch', 'scipy'],
)