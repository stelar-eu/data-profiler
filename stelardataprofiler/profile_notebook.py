"""Functionality related to displaying the profile report in Jupyter notebooks."""
import html
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from IPython.core.display import HTML
    from IPython.lib.display import IFrame

from pandas_profiling.config import IframeAttribute, Settings


def __get_notebook_iframe(
        config: Settings, html_doc: str
) -> Union["IFrame", "HTML"]:
    """Display the profile report in an iframe in the Jupyter notebook
    Args:
        config: Settings
        html_doc: The profile report object
    Returns:
        Displays the Iframe
    """

    attribute = config.notebook.iframe.attribute
    if attribute == IframeAttribute.srcdoc:
        output = __get_notebook_iframe_srcdoc(config, html_doc)
    else:
        raise ValueError(
            f'Iframe Attribute can omly be "srcdoc" (current: {attribute}).'
        )

    return output


def __get_notebook_iframe_srcdoc(config: Settings, html_doc: str) -> "HTML":
    """Get the IPython HTML object with iframe with the srcdoc attribute
    Args:
        config: Settings
        html_doc: The profile report object
    Returns:
        IPython HTML object.
    """
    from IPython.core.display import HTML

    width = config.notebook.iframe.width
    height = config.notebook.iframe.height
    src = html.escape(html_doc)

    iframe = f'<iframe width="{width}" height="{height}" srcdoc="{src}" frameborder="0" allowfullscreen></iframe>'

    return HTML(iframe)
