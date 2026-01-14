import argparse
import numpy as np


def str2bool(v):
    """
    Parse a string into a boolean value.

    Args:
        v (str or bool): Input value to interpret.

    Returns:
        bool: Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input is not a valid boolean string.
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()
