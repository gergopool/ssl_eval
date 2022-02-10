from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

setup(
    name='ssl_eval',
    version='0.0.1',
    author='Gergely Papp',
    author_email='gergopool@gmail.com',
    packages=find_packages(),
    package_dir={'ssl_eval': 'ssl_eval'},
    package_data={'ssl_eval': ['res/*.txt']},
    url='https://github.com/gergopool/ssl_eval',
    license='LICENSE',
    description='A plug & play evaluator for self-supervised image classification.',
    install_requires=requirements,
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)