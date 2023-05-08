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
    description='dataprofiler is a component providing functionalities for profiling different types of files and data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Panagiotis Betchavas',
    author_email='pbetchavas@athenarc.gr',
    python_requires='>=3.8, <4.0',
    packages=['stelardataprofiler', 'stelardataprofiler.variables'],
    data_files=[('json_files', ['stelardataprofiler/json_files/tsfresh_json.json']),
                ('mappings', ['stelardataprofiler/mappings/tabular_mapping.ttl', 
                'stelardataprofiler/mappings/raster_mapping.ttl', 
                'stelardataprofiler/mappings/hierarchical_mapping.ttl',
                'stelardataprofiler/mappings/rdfgraph_mapping.ttl', 
                'stelardataprofiler/mappings/textual_mapping.ttl'])],
    package_dir={
        'stelardataprofiler.variables': 'stelardataprofiler/variables',
    },
    url='https://github.com/stelar-eu/data-profiler',
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        'console_scripts':[
            'profiler-mappings = stelardataprofiler.main:main',
        ]
    }
)