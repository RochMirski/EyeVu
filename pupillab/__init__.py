"""pupillab — an extensible pupil-detection lab for EyeVu.

A small registry framework that lets several pupil detectors run side-by-side on
the same capture, each contributing its own labelled debug stages so the montage
(and the Streamlit dashboard) extend automatically as modules are added.

The production detector still lives in ``cap.py`` and is reused unchanged — the
``ridge_baseline`` module simply wraps it.  Nothing here is imported by the Pi
code, so this package is purely a dev-machine experimentation surface.

Typical use:
    from pupillab import registry, context
    registry.discover()                       # import all modules -> they register
    ctx = context.build_context(amb, flash, both, meta)
    for det in registry.get_all():
        result = det.detect(ctx, det.default_params())
"""

from . import base, registry, context  # noqa: F401

__all__ = ["base", "registry", "context"]
