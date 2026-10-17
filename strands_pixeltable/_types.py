"""Parsing of the ``type_`` strings in ``Table.get_metadata()``."""


def column_base(type_: object) -> str:
    """Base type name, e.g. ``'Array'`` for ``'Array[(8,), Float] | None'``."""
    return str(type_).split(" | ", 1)[0].split("[", 1)[0]


def is_nullable_type(type_: object) -> bool:
    return str(type_).endswith(" | None")
