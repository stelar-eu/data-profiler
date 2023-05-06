from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name='dataprofiler-st',
    version='0.1.0',
    description='dataprofiler is a component providing functionalities for profiling different types of data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Panagiotis Betchavas',
    author_email='pbetchavas@athenarc.gr',
    packages=['stelardataprofiler', 'stelardataprofiler.variables'],
    data_files=[('json_files', ['stelardataprofiler/json_files/tsfresh_json.json'])],
    package_dir={
        'stelardataprofiler.variables': 'stelardataprofiler/variables',
    },
    url='https://github.com/stelar-eu/data-profiler',
    python_requires='>=3.8, <4.0',
    install_requires=requirements,
    include_package_data=True
)
