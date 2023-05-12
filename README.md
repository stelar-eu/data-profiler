
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
Change the [config_template](https://github.com/stelar-eu/data-profiler/blob/main/config_template.json) according to the requirements of each profiler and execute [main.py](https://github.com/stelar-eu/data-profiler/blob/main/stelardataprofiler/main.py) to create the mapping.ttl file.

### Execute profiler-mappings script (after local library installation)

```sh
cd data-profiler
profiler-mappings config_template.json
```
> **_NOTE:_**  We can execute profile-mappings from anywhere as it is a console script, but we must have the correct path to the config_template.json and change the 'path' parameters of the config_template.json to correctly take the input and write the output.

### Profiler output
#### JSON
All profiling functions output the results in a JSON and an HTML file. A brief example of the JSON output of the raster profiler given two images as input is as follows.

```
{
"analysis":  { "date_start": "2023-04-28 12:09:45.815132",
               "date_end": "2023-04-28 12:09:54.920661",
                ... 
             },
"table":     { "byte_size": 2925069,
               "n_of_imgs": 2,
                ...
             },
"variables": \[{"name": "image_1",
               "type": "Raster",
               "crs": "EPSG:4326",
               "spatial_coverage": "POLYGON ((83 275, 183 0, 83 275))"
              }, ...\]
}
```

In short, the ```analysis``` field contains some metadata regarding the profiling task, such as the start and end time. The ```table``` field contains profiling results regarding the whole dataset, i.e., not considering the input images separately (e.g., number of images and total size in bytes). Finally, the ```variables``` field contains per image results, such as the CRS and spatial coverage.

A complete JSON output example can be found [here](https://github.com/stelar-eu/data-profiler/blob/main/examples/output/tabular_vector_profile.json).

#### HTML
The HTML file contains various plots that visualize the profiling results. An example can be found [here](https://github.com/stelar-eu/data-profiler/blob/main/examples/output/tabular_vector_profile.html).


### Apply mappings to generate RDF graph

Predefined mappings for profiles of the various types of datasets can be used to generate an RDF graph with the profiling information. 
The customized mappings are expressed in the RDF Mapping language (RML) and can be used to transform the JSON profile into various serializations in RDF, as specified by the user in a configuration.
To apply such mappings, you need to download the latest release of [RML Mapper](https://github.com/RMLio/rmlmapper-java/releases/) and execute the downloaded JAR in Java as follows: 

```sh
java -jar <path-to-RML_Mapper.JAR> -m <output-path>/mapping.ttl -d -s <RDF-serialization> -o <path-to-output-RDF-file>
```

File ```mapping.ttl``` required for this step has been created in the same folder as the JSON output produced by the data-profiler, as specified in the user's configuration. 
Options for the ```<RDF-serialization>``` include: ```nquads``` (__default__), ```turtle```, ```ntriples```, ```trig```, ```trix```, ```jsonld```, ```hdt```. If the path to the output RDF file is ommitted, then the RDF triples will be listed in standard output.

> **_NOTE:_**  Executing this operation with the RML Mapper requires Java 11 or later.

### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/stelar-eu/data-profiler/blob/main/LICENSE).
