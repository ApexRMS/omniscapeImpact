## omniscapeImpact

import pysyncrosim as ps
import numpy as np
import sys

from constants import NODATA_VALUE, GRID_TOLERANCE_FRACTION

# Helper functions -------------------------------------------------------------

def safeProgressBar(message, report_type = "message"):
    """Report progress to SyncroSim, falling back to the console.

    ps.environment.progress_bar calls _validate_environment internally, which
    raises RuntimeError whenever the SSIM_* environment variables are absent -
    that is, any time this script is run outside a real SyncroSim invocation.
    Wrapping it means the transformer can be run and debugged standalone.
    """
    try:
        ps.environment.progress_bar(message = message, report_type = report_type)
    except RuntimeError:
        print("[Progress] " + str(message))


def safeUpdateRunLog(*message, sep = "", type = "status"):
    """Write to the SyncroSim run log, falling back to the console.

    ps.environment.update_run_log validates the SyncroSim environment in the
    same way as progress_bar, so it fails in the same circumstances. omniscape
    wraps only progress_bar and leaves this one bare; both are wrapped here so
    that a standalone run completes rather than failing at the first log line.
    """
    try:
        ps.environment.update_run_log(*message, sep = sep, type = type)
    except RuntimeError:
        print("[Run log] " + sep.join(str(m) for m in message))


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
    meaningful if both rasters cover the same ground, at the same resolution, in
    the same coordinate system. Two rasters can share dimensions while sitting
    over entirely different terrain, so all three properties are checked.

    The affine transform is compared with a sub-pixel tolerance rather than for
    exact equality. A raster merged from spatial tiles can differ from one
    written in a single pass in the last floating-point digit with no
    consequence for the analysis, but a genuine offset of even a single pixel is
    far larger than the tolerance and is caught.
    """
    # Dimensions
    if baseSource.shape != altrSource.shape:
        sys.exit(
            "The Baseline and Alternative '" + rasterLabel + "' rasters have "
            "different dimensions (" + repr(baseSource.shape) + " and "
            + repr(altrSource.shape) + "). Both Scenarios must be run over the "
            "same extent and resolution before they can be compared.")

    # Coordinate reference system
    if baseSource.crs != altrSource.crs:
        sys.exit(
            "The Baseline and Alternative '" + rasterLabel + "' rasters use "
            "different coordinate reference systems (" + repr(baseSource.crs)
            + " and " + repr(altrSource.crs) + "). Both Scenarios must use the "
            "same projection before they can be compared.")

    # Affine transform, to within a fraction of one pixel
    tolerance = GRID_TOLERANCE_FRACTION * min(
        abs(baseSource.res[0]), abs(baseSource.res[1]))
    baseTransform = list(baseSource.transform)[:6]
    altrTransform = list(altrSource.transform)[:6]

    if any(abs(b - a) > tolerance for b, a in zip(baseTransform, altrTransform)):
        sys.exit(
            "The Baseline and Alternative '" + rasterLabel + "' rasters are not "
            "aligned to the same grid. Their pixel origins or resolutions differ "
            "by more than " + repr(tolerance) + " map units (Baseline "
            + repr(baseTransform) + ", Alternative " + repr(altrTransform)
            + "). Both Scenarios must be run over the same extent and "
            "resolution before they can be compared.")


def sameCategoryThresholds(baseThresholds, altrThresholds):
    """Return True if both Scenarios classify connectivity the same way.

    'Category Thresholds' are set per Scenario, so two Scenarios can legitimately
    use different cut-offs. When they do, their connectivity category rasters are
    built on different definitions, and comparing them would measure the change
    in definition rather than a change in connectivity. The 'Normalized current'
    comparison is unaffected, because it never uses thresholds.
    """
    if baseThresholds.empty | altrThresholds.empty:
        return False

    thresholdColumns = ["movementType", "minValue", "maxValue"]

    if not set(thresholdColumns).issubset(baseThresholds.columns):
        return False

    if not set(thresholdColumns).issubset(altrThresholds.columns):
        return False

    # Row order carries no meaning, so compare the sorted set of thresholds
    baseSorted = baseThresholds[thresholdColumns].sort_values(
        by = thresholdColumns).reset_index(drop = True)
    altrSorted = altrThresholds[thresholdColumns].sort_values(
        by = thresholdColumns).reset_index(drop = True)

    return baseSorted.equals(altrSorted)


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