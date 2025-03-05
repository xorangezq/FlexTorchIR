from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

__version__ = '1.0.0'

setup(
    name='FlexTorchIR',
    version=__version__,
    description='fast swissknife for disecting pytorch IR',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author='oliverxu',
    author_email='oliver_career@foxmail.com',

    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Customer Service',

        'Natural Language :: English',
        'Natural Language :: Chinese (Simplified)',
    ],

    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=requirements,
    include_package_data=True,
)
