"""Detector modules — importing this package registers every detector.

To add a detector: create ``my_detector.py`` here with a ``@register``-decorated
``PupilDetector`` subclass, then add one import line below.  ``registry.discover()``
imports this package, so the new module then appears in the dashboard, the
montage, and the batch runner with no further wiring.
"""

from . import ridge_baseline   # noqa: F401  — the proven cap.py detector (baseline)
from . import bonteanu_hough   # noqa: F401  — circular Hough (Bonteanu, literature)
from . import ritnet_seg       # noqa: F401  — pre-trained RITnet segmentation (ML)

__all__ = ["ridge_baseline", "bonteanu_hough", "ritnet_seg"]
