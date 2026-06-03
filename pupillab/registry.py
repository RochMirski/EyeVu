"""Detector registry.

Modules register themselves by decorating their class with ``@register``.
``discover()`` imports the ``pupillab.modules`` package, which imports every
module file, so a single ``discover()`` call populates the registry.  Adding a
new detector is therefore: write a file under ``modules/``, decorate the class,
add it to ``modules/__init__.py`` — done.
"""

from __future__ import annotations

from typing import Iterable

from .base import PupilDetector

# Insertion-ordered: dashboards/montages list detectors in registration order.
REGISTRY: "dict[str, PupilDetector]" = {}

_discovered = False


def register(cls):
    """Class decorator: instantiate the detector and add it to the registry."""
    instance = cls()
    if not getattr(instance, "name", None) or instance.name == "unnamed":
        raise ValueError(f"{cls.__name__} must set a unique `name`")
    if instance.name in REGISTRY:
        raise ValueError(f"duplicate detector name: {instance.name!r}")
    REGISTRY[instance.name] = instance
    return cls


def discover() -> "dict[str, PupilDetector]":
    """Import all module files so they self-register; return the registry."""
    global _discovered
    if not _discovered:
        from . import modules  # noqa: F401  (its __init__ imports every module)
        _discovered = True
    return REGISTRY


def get_all() -> "list[PupilDetector]":
    return list(discover().values())


def get(name: str) -> PupilDetector:
    return discover()[name]


def names() -> Iterable[str]:
    return discover().keys()
