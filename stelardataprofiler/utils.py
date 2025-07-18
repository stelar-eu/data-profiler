import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Any


def read_config(json_file: str) -> dict:
    """
    This method reads configuration settings from a json file. Configuration includes all parameters for input/output.

    :param json_file: path to .json file that contains the configuration parameters.
    :type json_file: str
    :return: A dictionary with all configuration settings.
    :rtype: dict

    """
    try:
        config_dict: dict = json.loads(json_file)
    except ValueError as e:
        with open(json_file) as f:
            config_dict: dict = json.load(f)
            return config_dict

    return config_dict


def write_to_json(output_dict: dict, output_file: Union[str, Path]) -> None:
    """
    Write the profile dictionary to a file.

    :param output_dict: the profile dictionary that will writen.
    :type output_dict: dict
    :param output_file: The name or the path of the file to generate including the extension (.json).
    :type output_file: Union[str, Path]
    :return: a dict which contains the results of the profiler for the texts.
    :rtype: dict

    """
    if not isinstance(output_file, Path):
        output_file = Path(str(output_file))

    # create image folder if it doesn't exist
    path = Path(str(output_file.parent))
    path.mkdir(parents=True, exist_ok=True)

    if output_file.suffix == ".json":
        with open(output_file, "w") as outfile:
            def encode_it(o: Any) -> Any:
                if isinstance(o, dict):
                    return {encode_it(k): encode_it(v) for k, v in o.items()}
                else:
                    if isinstance(o, (bool, int, float, str)):
                        return o
                    elif isinstance(o, list):
                        return [encode_it(v) for v in o]
                    elif isinstance(o, set):
                        return {encode_it(v) for v in o}
                    elif isinstance(o, (pd.DataFrame, pd.Series)):
                        return encode_it(o.reset_index().to_dict("records"))
                    elif isinstance(o, np.ndarray):
                        return encode_it(o.tolist())
                    elif isinstance(o, np.generic):
                        return o.item()
                    else:
                        return str(o)

            output_dict = encode_it(output_dict)
            json.dump(output_dict, outfile, indent=3)
    else:
        suffix = output_file.suffix
        warnings.warn(
            f"Extension {suffix} not supported. For now we assume .json was intended. "
            f"To remove this warning, please use .json."
        )
