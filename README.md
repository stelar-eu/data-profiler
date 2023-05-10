
## data-profiler

### Overview

data-profiler is a Python library providing various functions for profiling different types of data and files.

### Quick start

Please see the provided [notebooks](https://github.com/stelar-eu/data-profiler/tree/main/notebooks).

### Documentation

Please see [here](https://stelar-eu.github.io/data-profiler/).

### Installation
data-profiler needs python version >=3.8 and < 4.0.

#### Python Module - Local library

data-profiler, after it is downloaded from [here](https://github.com/stelar-eu/data-profiler) can be installed with:

```sh
$ cd data-profiler
$ pip install .
```
#### How to import local library

After you install the data-profile as a local library you can import it in your python:

```python
import stelardataprofiler
```

### Configuration
Change the [config](https://github.com/stelar-eu/data-profiler/blob/main/config.json) according to the requirements of each profiler and execute [main.py](https://github.com/stelar-eu/data-profiler/blob/main/stelardataprofiler/main.py) to create the mapping.ttl file.

### Execute profiler-mappings script (after local library installation)

```sh
cd data-profiler
profiler-mappings config.json
```
> **_NOTE:_**  We can execute profile-mappings from anywhere as it is a console script, but we must have the correct path to the config.json and change the 'path' parameters of the config.json to correctly take the input and write the output.

### Apply mappings to generate RDF graph

Predefined mappings for profiles of the various types of datasets can be used to generate an RDF graph with the profiling information. 
The customized mappings are expressed in the RDF Mapping language (RML) and can be used to transform the JSON profile into various serializations in RDF, as specified by the user in a configuration.
To apply such mappings, you need to download the latest release of [RML Mapper](https://github.com/RMLio/rmlmapper-java/releases/) and execute the downloaded JAR in Java as follows: 

```sh
java -jar <path-to-RML_Mapper.JAR> -m <output-path>/mapping.ttl -d -s <RDF-serialization> -o <path-to-output-RDF-file>
```

Note that the required ```mapping.ttl``` will be located in the same folder as the JSON profile, as specified in the user's configuration. 
Options for the ```<RDF-serialization>``` include: ```nquads``` (__default__), ```turtle```, ```ntriples```, ```trig```, ```trix```, ```jsonld```, ```hdt```. If the path to the output RDF file is ommitted, then the RDF triples will be listed in standard output.

### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/stelar-eu/data-profiler/blob/main/LICENSE).