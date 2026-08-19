"""Pure-function core of the mosaicing pipeline.

No Pi-specific IO, no camera access, no GPIO, no hardcoded paths.  Every module
here takes data and a `MosaicConfig` and returns data.
"""

from . import (bundle, compose, features, globalopt, graph, models, pairwise,
               preprocess)

__all__ = ["bundle", "compose", "features", "globalopt", "graph", "models",
           "pairwise", "preprocess"]
