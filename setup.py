from setuptools import setup, find_packages

setup(
    name='pyktlib',
    version='0.1.2',
    author='ZhijieXiong',
    author_email='xzjdream@gmail.com',
    description='python lib for knowledge tracing model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AI-for-Education-free/pyktlib',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[],
)
