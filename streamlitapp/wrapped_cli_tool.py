import click
import streamlit.web.cli as stcli
import os
from streamlit import config as _config
from streamlit.config_option import ConfigOption
import streamlit.web.bootstrap as bootstrap
from typing import Any, Dict

def _convert_config_option_to_click_option(
    config_option: ConfigOption,
) -> Dict[str, Any]:
    """Composes given config options as options for click lib."""
    option = f"--{config_option.key}"
    param = config_option.key.replace(".", "_")
    description = config_option.description
    if config_option.deprecated:
        if description is None:
            description = ""
        description += (
            f"\n {config_option.deprecation_text} - {config_option.expiration_date}"
        )

    return {
        "param": param,
        "description": description,
        "type": config_option.type,
        "option": option,
        "envvar": config_option.env_var,
    }


def _make_sensitive_option_callback(config_option: ConfigOption):
    def callback(_ctx: click.Context, _param: click.Parameter, cli_value) -> None:
        if cli_value is None:
            return None
        raise SystemExit(
            f"Setting {config_option.key!r} option using the CLI flag is not allowed. "
            f"Set this option in the configuration file or environment "
            f"variable: {config_option.env_var!r}"
        )

    return callback


def configurator_options(func):
    """Decorator that adds config param keys to click dynamically."""
    for _, value in reversed(_config._config_options_template.items()):
        parsed_parameter = _convert_config_option_to_click_option(value)
        if value.sensitive:
            # Display a warning if the user tries to set sensitive
            # options using the CLI and exit with non-zero code.
            click_option_kwargs = {
                "expose_value": False,
                "hidden": True,
                "is_eager": True,
                "callback": _make_sensitive_option_callback(value),
            }
        else:
            click_option_kwargs = {
                "show_envvar": True,
                "envvar": parsed_parameter["envvar"],
            }
        config_option = click.option(
            parsed_parameter["option"],
            parsed_parameter["param"],
            help=parsed_parameter["description"],
            type=parsed_parameter["type"],
            **click_option_kwargs,
        )
        func = config_option(func)
    return func


@click.group()
def main():
    pass

@main.command("run")
@configurator_options
@click.argument("args", nargs=-1)
def main_streamlit(args=None, **kwargs):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'app.py')
    bootstrap.load_config_options(flag_options=kwargs)

    stcli._main_run(filename, args, flag_options=kwargs)


if __name__ == "__main__":
    main()
