## omniscapeImpact

import pysyncrosim as ps
import pandas as pd
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
            "The '" + rasterLabel + "' rasters being compared have "
            "different dimensions (" + repr(baseSource.shape) + " and "
            + repr(altrSource.shape) + "). Both Scenarios must be run over the "
            "same extent and resolution before they can be compared.")

    # Coordinate reference system
    if baseSource.crs != altrSource.crs:
        sys.exit(
            "The '" + rasterLabel + "' rasters being compared use "
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
            "The '" + rasterLabel + "' rasters being compared are not "
            "aligned to the same grid. Their pixel origins or resolutions differ "
            "by more than " + repr(tolerance) + " map units (Baseline "
            + repr(baseTransform) + ", Alternative " + repr(altrTransform)
            + "). Both Scenarios must be run over the same extent and "
            "resolution before they can be compared.")


def chooseComparisonScenarios(dependencyTable):
    """Choose the Baseline and Alternative Scenarios from a dependency table.

    The Scenarios to compare are read from the Scenario's dependencies rather
    than typed in by ID. The table is the DataFrame returned by pysyncrosim's
    Scenario.dependencies property (columns Id, Name, Priority, ordered by
    priority, where priority 1 is the first dependency added).

    Convention: the FIRST dependency is the Baseline and the SECOND is the
    Alternative. If more than two dependencies are present, the first two are
    used and the rest are ignored with a warning.

    Returns (baselineId, alternativeId, message) where message describes the
    choice for the run log, including any ignored dependencies.
    """
    nDependencies = len(dependencyTable)

    if nDependencies < 2:
        found = ("none were found." if nDependencies == 0 else
                 "only 1 was found: '" + str(dependencyTable.Name.iloc[0]) + "'.")
        sys.exit(
            "The Connectivity Impact Assessment requires exactly 2 Scenario "
            "dependencies, but " + found + " Add the two omniscape Scenarios to "
            "compare as dependencies of this Scenario: first the Baseline, then "
            "the Alternative.")

    ordered = dependencyTable.sort_values(by = "Priority").reset_index(drop = True)

    baselineId = int(ordered.Id.iloc[0])
    alternativeId = int(ordered.Id.iloc[1])

    message = ("Comparing dependencies: Baseline = '" + str(ordered.Name.iloc[0])
               + "' (Scenario ID " + repr(baselineId) + "), Alternative = '"
               + str(ordered.Name.iloc[1]) + "' (Scenario ID "
               + repr(alternativeId) + "). To swap them, reorder the "
               "dependencies.")

    if nDependencies > 2:
        ignoredNames = ", ".join("'" + str(n) + "'" for n in ordered.Name.iloc[2:])
        message += (" WARNING: " + repr(nDependencies) + " dependencies were "
                    "found but only the first 2 are compared. Ignored: "
                    + ignoredNames + ".")

    return baselineId, alternativeId, message


def resolveConnectivitySurface(omniscapeOutput, ensembleOutput, scenarioLabel):
    """Choose which continuous connectivity surface represents a Scenario.

    A Scenario reaches this package by one of two routes. Run through
    omniscape's 'Omniscape' transformer it produces a 'Normalized current'
    raster; run through 'Ensemble Connectivity' it produces an ensemble raster
    combining several other Scenarios. Either can then be categorized, so
    either can be compared - but only against the same kind of surface.

    The ensemble wins when both are present, matching how omniscape's own
    'Categorize Connectivity Output' transformer chooses: a Scenario that ran
    the ensemble is asking for its combined surface to be used, not whatever
    single-model output happens to sit alongside it.

    Returns (path, kind, label) where kind is "ensemble" or "normalizedCurrent"
    and label is the datasheet column's display name, for messages.
    """
    ensemblePath = firstPopulatedValue(ensembleOutput, "ensembleRaster")

    if ensemblePath is not None:
        return str(ensemblePath), "ensemble", "Ensemble connectivity"

    normalizedPath = firstPopulatedValue(omniscapeOutput, "normalizedCumCurrmap")

    if normalizedPath is not None:
        return str(normalizedPath), "normalizedCurrent", "Normalized current"

    sys.exit(
        "No connectivity surface was found for the " + scenarioLabel + " Scenario. "
        "Run omniscape's 'Omniscape' transformer to produce a 'Normalized current' "
        "raster, or 'Ensemble Connectivity' to produce an 'Ensemble connectivity' "
        "raster, before comparing the Scenario.")


def firstPopulatedValue(datasheet, column):
    """Return a populated value from a single-row output datasheet, or None.

    A single-row output datasheet that a transformer never wrote comes back
    either with no rows at all or with the column present but null, and which
    of the two you get depends on the Scenario's history rather than on
    anything meaningful. Both mean the same thing here.
    """
    if datasheet is None or datasheet.empty or column not in datasheet.columns:
        return None

    value = datasheet[column].iloc[0]

    if value is None or value != value:               # NaN
        return None

    return value


def validateComparableSurfaces(baseKind, altrKind, baseLabel, altrLabel):
    """Exit unless the two Scenarios' connectivity surfaces are the same kind.

    An ensemble surface and a single-model normalized current measure different
    quantities: one is a weighted combination across several Scenarios, on a
    scale set by how many went into it and how they were weighted, the other is
    one model's current normalized against its own flow potential. Subtracting
    one from the other returns a number for every pixel, but that number is the
    difference between two different measurements rather than the impact of an
    intervention - and nothing downstream could tell the two apart.
    """
    if baseKind == altrKind:
        return

    sys.exit(
        "The two Scenarios being compared produced different kinds of "
        "connectivity surface: the Baseline has '" + baseLabel + "' and the "
        "Alternative has '" + altrLabel + "'. These measure different "
        "quantities, so the difference between them would not describe an "
        "impact. Compare two ensemble Scenarios with each other, or two "
        "single-model Scenarios with each other.")


def validateOneRowPerCategory(tabularSummary, scenarioLabel):
    """Exit unless each connectivity category appears exactly once.

    omniscape's 'Connectivity Categories Summary' is expected to hold one row
    per category. If a category appears more than once there is no safe way to
    combine the rows: summing is only correct for area if the duplicates are
    disjoint partial counts, and proportions cannot be summed at all because
    each would be relative to a different denominator. Rather than guess, this
    reports what was found.
    """
    duplicated = tabularSummary.movementTypesID[
        tabularSummary.movementTypesID.duplicated()].unique()

    if len(duplicated) > 0:
        sys.exit(
            "The " + scenarioLabel + " Scenario's 'Connectivity Categories "
            "Summary' contains more than one row for " + repr(len(duplicated))
            + " connectivity category or categories (" + ", ".join(
                repr(d) for d in duplicated) + "), across " + repr(len(tabularSummary))
            + " rows in total. Each connectivity category must appear exactly "
            "once. This can occur if the Scenario was run using spatial "
            "multiprocessing.")


def alignCategorySummaries(baseTabular, altrTabular):
    """Join the two Scenarios' category summaries on category, not row order.

    The summaries are joined on 'movementTypesID' rather than being subtracted
    positionally. omniscape omits any category that occupies no pixels, so the
    two Scenarios can legitimately contain different categories, in different
    orders, and a positional subtraction would silently attribute differences
    to the wrong categories.

    A category missing from one Scenario means it occupies no pixels there,
    which is a measured result rather than missing information, so it is filled
    with zero. This is what allows a category disappearing entirely to be
    reported as a total loss instead of as NaN.
    """
    validateOneRowPerCategory(baseTabular, "Baseline")
    validateOneRowPerCategory(altrTabular, "Alternative")

    summaryColumns = ["movementTypesID", "amountArea", "percentCover"]

    merged = baseTabular[summaryColumns].merge(
        altrTabular[summaryColumns],
        on = "movementTypesID",
        how = "outer",
        suffixes = ("Base", "Altr"),
        validate = "one_to_one")

    # A category absent from a Scenario covers no area and no proportion of it
    merged = merged.fillna({"amountAreaBase": 0.0, "percentCoverBase": 0.0,
                            "amountAreaAltr": 0.0, "percentCoverAltr": 0.0})

    return pd.DataFrame({
        "movementTypesID": merged.movementTypesID,
        "amountAreaDifference": merged.amountAreaAltr - merged.amountAreaBase,
        "percentCoverDifference": merged.percentCoverAltr - merged.percentCoverBase})


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


def sameCategoryBreaks(baseTabular, altrTabular):
    """Compare the break values each Scenario actually used to categorize.

    'Category Thresholds' record what was *requested*; omniscape 2.8's
    minBreakValue / maxBreakValue record what those thresholds *worked out to*.
    The two diverge whenever Threshold type is Quantile, because a quantile is
    a position in the Scenario's own distribution rather than a fixed value. A
    quantile of 0.9 means "the top tenth of this Scenario", so two Scenarios
    asking for identical quantiles get identical threshold datasheets and
    entirely different cut-offs.

    That matters most for exactly the case quantiles were added for. Categories
    defined by rank rather than value move with the surface they describe, so a
    'High' category stays the top tenth of the landscape however much
    connectivity the intervention removed, and the comparison reports a change
    close to zero no matter what happened.

    Only the categories present in both Scenarios are compared: a category
    absent from one occupies no pixels there, which says nothing about whether
    the two agree on how connectivity was cut up.

    Returns (comparable, reason). comparable is None when the break values are
    unavailable - a library written before omniscape 2.8, whose Scenarios could
    only have used Value mode - leaving the caller to fall back to comparing the
    requested thresholds.
    """
    breakColumns = ["minBreakValue", "maxBreakValue"]

    if baseTabular is None or altrTabular is None:
        return None, None

    if baseTabular.empty or altrTabular.empty:
        return None, None

    if not set(breakColumns).issubset(baseTabular.columns):
        return None, None

    if not set(breakColumns).issubset(altrTabular.columns):
        return None, None

    keyColumns = ["movementTypesID"] + breakColumns
    baseBreaks = baseTabular[keyColumns].dropna()
    altrBreaks = altrTabular[keyColumns].dropna()

    if baseBreaks.empty or altrBreaks.empty:
        return None, None

    shared = baseBreaks.merge(altrBreaks, on = "movementTypesID",
                              suffixes = ("Base", "Altr"))

    if shared.empty:
        return None, None

    # Both sides are computed from raster data, so compare within floating-point
    # tolerance rather than for exact equality
    differing = shared[
        ~(np.isclose(shared.minBreakValueBase, shared.minBreakValueAltr,
                     rtol = 1e-9, atol = 1e-12)
          & np.isclose(shared.maxBreakValueBase, shared.maxBreakValueAltr,
                       rtol = 1e-9, atol = 1e-12))]

    if differing.empty:
        return True, None

    example = differing.iloc[0]
    reason = (
        "The Baseline and Alternative Scenarios categorized connectivity at "
        "different break values, so their connectivity categories do not "
        "describe the same ranges. Connectivity category " + repr(int(example.movementTypesID))
        + ", for instance, covers " + repr(float(example.minBreakValueBase)) + " to "
        + repr(float(example.maxBreakValueBase)) + " in the Baseline but "
        + repr(float(example.minBreakValueAltr)) + " to "
        + repr(float(example.maxBreakValueAltr)) + " in the Alternative ("
        + repr(len(differing)) + " of " + repr(len(shared)) + " shared categories "
        "differ). This is what happens when 'Threshold type' is set to Quantile: "
        "each Scenario's breaks are computed from its own distribution, so the "
        "categories move with the surface and a comparison between them would "
        "understate the impact. Set 'Threshold type' to Value in both Scenarios, "
        "using the same 'Category Thresholds', to compare connectivity categories.")

    return False, reason


def categoriesAreComparable(baseTabular, altrTabular, baseThresholds, altrThresholds):
    """Decide whether the two Scenarios' connectivity categories can be compared.

    Prefers the break values actually used, and falls back to the requested
    'Category Thresholds' when those are not recorded. The realized breaks are
    the stronger test: identical thresholds do not imply identical breaks under
    Quantile mode, while identical breaks mean the categories cut the surface at
    the same places however they were specified.

    Returns (comparable, reason), where reason explains a False for the run log.
    """
    comparable, reason = sameCategoryBreaks(baseTabular, altrTabular)

    if comparable is not None:
        return comparable, reason

    if sameCategoryThresholds(baseThresholds, altrThresholds):
        return True, None

    if baseThresholds.empty or altrThresholds.empty:
        return False, (
            "No 'Category Thresholds' were recorded for one or both Scenarios, "
            "so there is no way to confirm that their connectivity categories "
            "describe the same ranges.")

    return False, (
        "The Baseline and Alternative Scenarios use different 'Category "
        "Thresholds'. Connectivity categories are therefore not comparable "
        "between them.")


def reportNodataFootprint(baseMask, altrMask, rasterLabel):
    """Report pixels that hold valid data in only one of the two Scenarios.

    The two Scenarios' valid areas are allowed to differ: comparisons are made
    over the pixels valid in BOTH (callers mask with the union of the two
    no-data masks), so a one-sided pixel is simply excluded rather than
    compared. It still deserves a warning, because differing coverage often
    means the Scenarios were run over different extents, with different
    resistance or source layers, or with different spatial tiling settings -
    and everything excluded is invisible in the outputs.

    Returns the number of one-sided pixels (0 when the footprints agree).
    """
    nDisagree = int((baseMask != altrMask).sum())

    if nDisagree > 0:
        nValidUnion = int((~baseMask | ~altrMask).sum())
        share = (100.0 * nDisagree / nValidUnion) if nValidUnion > 0 else 0.0
        safeUpdateRunLog(
            "WARNING: " + repr(nDisagree) + " pixels ("
            + ("%.1f" % share) + "% of the valid area) hold data in only one "
            "Scenario for the '" + rasterLabel + "' rasters. These pixels were "
            "excluded from all comparisons. Differing coverage usually means "
            "the two Scenarios were run over different extents, with different "
            "resistance or source layers, or with different spatial tiling "
            "settings.")

    return nDisagree