## omniscapeImpact

import pysyncrosim as ps
import pandas as pd
import os
import rasterio
import numpy as np
import itertools
import sys

from constants import NODATA_VALUE

# Helper functions -------------------------------------------------------------

def nodataMask(rasterSource, rasterData):
    """Return a boolean array that is True wherever a pixel holds no valid data.

    A raster can flag "no data" in more than one way, and as of omniscape 2.7.0
    which one you get depends on how the scenario happened to be run:

      * a sentinel value declared in the file header (normally -9999)
      * NaN, sometimes with no sentinel declared in the header at all, which is
        what spatial tiling produces when a tile carries no declared nodata
      * -9999 present in the pixels but not declared in the header

    All three are tested here so that masking does not depend on which omniscape
    code path produced the raster. Testing for -9999 unconditionally is safe for
    the rasters compared by this package: normalized current is a ratio and
    connectivity categories are small positive integers, so -9999 is never a
    legitimate value.
    """
    mask = np.zeros(rasterData.shape, dtype = bool)

    # Sentinel declared in the raster header, if there is one
    if rasterSource.nodata is not None:
        if np.isnan(rasterSource.nodata):
            mask |= np.isnan(rasterData)
        else:
            mask |= (rasterData == rasterSource.nodata)

    # NaN, whether or not the header declares it
    if np.issubdtype(rasterData.dtype, np.floating):
        mask |= np.isnan(rasterData)

    # This package's conventional sentinel, whether or not the header declares it
    mask |= (rasterData == NODATA_VALUE)

    return mask


def validateSameGrid(baseSource, altrSource, rasterLabel):
    """Exit unless the two rasters describe the same pixel grid.

    Every comparison this package makes is pixel-by-pixel, which is only
    meaningful if both rasters cover the same ground at the same resolution.
    """
    if baseSource.shape != altrSource.shape:
        sys.exit(
            "The Baseline and Alternative '" + rasterLabel + "' rasters have "
            "different dimensions (" + repr(baseSource.shape) + " and "
            + repr(altrSource.shape) + "). Both Scenarios must be run over the "
            "same extent and resolution before they can be compared.")


def validateNodataFootprint(baseMask, altrMask, rasterLabel):
    """Exit unless the two rasters agree on which pixels hold valid data.

    A disagreement means the two Scenarios do not cover the same valid area, so
    any comparison between them would be measuring a change in coverage as
    though it were a change in connectivity.
    """
    nDisagree = int((baseMask != altrMask).sum())

    if nDisagree > 0:
        sys.exit(
            "The Baseline and Alternative '" + rasterLabel + "' rasters "
            "disagree about which pixels hold valid data (" + repr(nDisagree)
            + " pixels differ). Both Scenarios must cover the same valid area "
            "before they can be compared. This usually means the two Scenarios "
            "were run over different extents, with different resistance or "
            "source layers, or with different spatial tiling settings.")