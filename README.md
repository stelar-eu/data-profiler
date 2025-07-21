# stelardataprofiler

stelardataprofiler is a Python library providing various functions for profiling different types of data and files.

## Quick start

Please see the provided [notebooks](https://github.com/stelar-eu/data-profiler/tree/main/notebooks).

## Documentation

Please see [here](https://stelar-eu.github.io/data-profiler/).

## Type Detection - Customize the profiler

In tabular and timeseries, the profiler automatically analyzes each column in the 
input data and assigns one of the following eight supported data types. 
The result can be stored either in a dictionary or a JSON file, which the user 
can review and modify if needed. The user may provide the modified dictionary or 
JSON file and execute an enhanced (more user-controlled) profiling task.

### Automatically Detected Data Types

| Data Type       | Description                                                  | Required Parameters   |
|-----------------|--------------------------------------------------------------|-----------------------|
| **Unsupported** | Type is not currently supported                              | None                  |
| **DateTime**    | Date or time-related values                                  | None                  |
| **Geometry**    | Geospatial data (points, shapes)                             | `crs`, `eps_distance` |
| **Categorical** | Discrete, labeled values                                     | None                  |
| **Textual**     | Free-form text data                                          | None                  |
| **Numeric**     | Numerical values (int/float)                                 | `max_freq_distr`      |
| **TimeSeries**  | Numerical values (int/float) based on time-indexed sequences | `max_freq_distr`      |
| **Boolean**     | True/False values (can also be 0 and 1 integers)             | None                  |

> ℹ️ **Note:** All required parameters have sensible default values and do not need to be explicitly set unless custom behavior is desired.

### Type Overrides

After detection, users can manually override the detected data types **as long as the change is semantically compatible**. This allows greater flexibility in how columns are processed during profiling.

#### ✅ Allowed Type Conversions

| From → To       | Compatible Changes (conversions with * may cause issues depending on the data) |
|-----------------|--------------------------------------------------------------------------------|
| **Numeric**     | Categorical, Textual, TimeSeries, Unsupported                                  |
| **TimeSeries**  | Categorical, Textual, Numeric, Unsupported                                     |
| **Categorical** | Textual, Numeric*, Unsupported                                                 |
| **Textual**     | Categorical, Unsupported                                                       |
| **Boolean**     | Numeric*, Categorical, Textual, Unsupported                                    |
| **Geometry**    | Categorical, Textual, Unsupported                                              |
| **DateTime**    | Categorical, Textual, Unsupported                                              |

> 🚫 Incompatible conversions (e.g., Boolean → Geometry, Categorical → Numeric 
> if we do not only have numeric data or Boolean → Numeric if we have true or false values 
> and not numeric data) are not allowed and may lead to errors or invalid outputs.

### Parameter Reference

| Parameter        | Used By             | Description                                                       |
|------------------|---------------------|-------------------------------------------------------------------|
| `max_freq_distr` | Numeric, TimeSeries | Maximum number of bins for frequency distribution visualizations  |
| `eps_distance`   | Geometry            | Distance tolerance for spatial clustering in geometry heatmaps    |
| `crs`            | Geometry            | Coordinate Reference System used for interpreting geospatial data |


## Installation
stelardataprofiler needs python version >=3.8 and < 3.13, also python version 
must not be 3.9.7.

### Python Module - Local library

stelardataprofiler can be installed with:

```sh
$ pip install stelardataprofiler
```
### How to import local library

After you install the stelardataprofiler as a local library you can import 
it in your python:

```python
import stelardataprofiler
```

### How to run the app

After you install the stelardataprofiler as a local library you can run the app 
by executing streamlit run inside the streamlitapp folder.

```sh
$ cd data-profiler/streamlitapp
$ streamlit run app.py
```

## Configuration
Change the [config_template](https://github.com/stelar-eu/data-profiler/blob/main/config_template.json) according to the requirements of each profiler 
and execute [main.py](https://github.com/stelar-eu/data-profiler/blob/main/stelardataprofiler/main.py) to create the mapping.ttl file.

## Execute profiler-mappings script 

```sh
profiler-mappings <absolute-folder-path>\config_template.json
```
> **_NOTE:_**  We can execute profile-mappings from anywhere as it is a console script, but we must have the correct path to the config_template.json and change the 'path' parameters of the config_template.json to correctly take the input and write the output.

## Output
### JSON
All profiling functions output the results in a JSON file. 
A brief example of the JSON output of the raster profiler given two images as input is as follows.

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
"variables": [{"name": "image_1",
               "type": "Raster",
               "crs": "EPSG:4326",
               "spatial_coverage": "POLYGON ((83 275, 183 0, 83 275))"
              }, ...]
}
```

In short, the ```analysis``` field contains some metadata regarding the profiling task, such as the start and end time. The ```table``` field contains profiling results regarding the whole dataset, i.e., not considering the input images separately (e.g., number of images and total size in bytes). Finally, the ```variables``` field contains per image results, such as the CRS and spatial coverage.

A complete JSON output example can be found [here](https://github.com/stelar-eu/data-profiler/blob/main/examples/output/tabular_vector_profile.json).

## Apply mappings to generate RDF graph

Predefined [mappings](https://github.com/stelar-eu/data-profiler/tree/main/stelardataprofiler/mappings) for profiles of the various types of datasets are available and can be used to generate an RDF graph with the profiling information. Once the profiling process completes, an automatically configured ```mapping.ttl``` file is available in the same folder as the output JSON.
All such customized mappings are expressed in the RDF Mapping language (RML) and can be used to transform the JSON profile into various serializations in RDF, as specified by the user in a configuration.
To apply such mappings, you need to download the latest release of [RML Mapper](https://github.com/RMLio/rmlmapper-java/releases/) and execute the downloaded JAR in Java as follows: 

```sh
java -jar <path-to-RML_Mapper.JAR> -m <output-path>/mapping.ttl -d -s <RDF-serialization> -o <path-to-output-RDF-file>
```

File ```mapping.ttl``` required for this step has been created in the same folder as the JSON output produced by the stelardataprofiler, as specified in the user's configuration. 
Options for the ```<RDF-serialization>``` include: ```nquads``` (__default__), ```turtle```, ```ntriples```, ```trig```, ```trix```, ```jsonld```, ```hdt```. If the path to the output RDF file is ommitted, then the RDF triples will be listed in standard output.

> **_NOTE:_**  Executing this operation with the RML Mapper requires Java 11 or later.

## License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/stelar-eu/data-profiler/blob/main/LICENSE).

## Acknowledgements

This work was partially funded by the EU Horizon Europe projects STELAR (GA. 101070122)
