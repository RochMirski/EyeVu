"""Post-session retinal mosaicing for EyeVu.

An importable package with a pure-function core: nothing under `core/` touches
a camera, GPIO, the network, or a hardcoded path, so the diagnostic dashboard on
the laptop imports and calls exactly the functions the Pi runs.

    from eyevu_mosaic import MosaicConfig, run_session
    result = run_session.run("Sessions/session_20260815_223238")

See README.md for the model ladder, the thresholds and measured timings.
"""

from .config import MosaicConfig, LadderConfig, DEFAULT

__all__ = ["MosaicConfig", "LadderConfig", "DEFAULT", "__version__"]
__version__ = "2.0.0"
