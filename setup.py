import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension


src = os.path.join("multi_mesh", 'src')
lib = Extension('multi_mesh',
                sources=[
                    os.path.join(src, "centroid.c"),
                    os.path.join(src, "trilinearinterpolator.c")],
                extra_compile_args=["-O3", "-fopenmp"],
                extra_link_args=['-lgomp'])


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='multi_mesh',
    version='0.1',
    author="The MultiMesh Development Team",
    long_description=readme(),
    packages=find_packages(),
    license="MIT",
    dependency_links=['https://github.com/SalvusHub/pyexodus'
                      '/archive/master.zip#egg=pyexodus-master'],
    install_requires=["numpy", "scipy", "click", "h5py", "pyexodus"],
    platforms="OS Independent",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'],
    entry_points='''
    [console_scripts]
    multi_mesh=multi_mesh.scripts.cli:cli
    ''',
    ext_package='multi_mesh.lib',
    ext_modules=[lib]
)
