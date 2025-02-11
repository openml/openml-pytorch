from typing import Iterable
import re
_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")

def listify(o=None) -> list:
    """
    Convert `o` to list. If `o` is None, return empty list.
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def camel2snake(name: str) -> str:
    """
    Convert `name` from camel case to snake case.
    """
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()