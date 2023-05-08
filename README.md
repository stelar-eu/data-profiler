
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


### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/stelar-eu/data-profiler/blob/main/LICENSE).