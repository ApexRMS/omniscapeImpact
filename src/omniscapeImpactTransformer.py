## omniscapeImpact

import pysyncrosim as ps
import pandas as pd
import os
import rasterio
import numpy as np
import itertools
import sys

from helperFunctions import (reportNodataFootprint, validateSameGrid, nodataMask,
                             categoriesAreComparable, alignCategorySummaries,
                             chooseComparisonScenarios, resolveConnectivitySurface,
                             validateComparableSurfaces, safeProgressBar, safeUpdateRunLog)
from constants import NODATA_VALUE

# Set up -----------------------------------------------------------------------

safeProgressBar(message="Setting up Scenario", report_type="message")

# Set environment and working directory
e = ps.environment._environment()
wrkDir = e.data_directory.item()

# Open SyncroSim Library, Project and Scenario
myLibrary = ps.Library()
myProject = myLibrary.projects(pid = 1) 
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)

if pd.isna(myScenario.parent_id):
    myParentScenario = myScenario  # Use self as parent if no parent exists
else:
    myScenarioParentID = int(myScenario.parent_id)
    myParentScenario = myLibrary.scenarios(sid = myScenarioParentID)

# Create directory, if applicable
outputCategoryPath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscapeImpact_outputSpatialCategory")
outputOverallPath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscapeImpact_outputSpatialOverall")
if os.path.exists(outputCategoryPath) == False:
    os.makedirs(outputCategoryPath)

if os.path.exists(outputOverallPath) == False:
    os.makedirs(outputOverallPath)


# Load input and settings from SyncroSim Library ------------------------------- 

# Input datasheets
movementTypeClasses = myProject.datasheets(name = "omniscape_movementTypes", include_key = True)



# Validation for inputs --------------------------------------------------------

if movementTypeClasses.empty:
    sys.exit("'Category Thresholds' are required.")



# Identify the Scenarios to compare from the dependencies -----------------------

# The two omniscape Scenarios to compare are supplied as dependencies of this
# Scenario (drag the Scenarios onto its Dependencies folder in SyncroSim
# Studio): first the Baseline, then the Alternative. Dependencies belong to the
# parent Scenario, so they are read from there.

dependencyTable = myParentScenario.dependencies

baselineID, alternativeID, dependencyMessage = chooseComparisonScenarios(dependencyTable)

safeUpdateRunLog(dependencyMessage)



# Open baseline and alternative scenario results -------------------------------

# Detect if Parent or Result Scenario IDs
allScenarios = myProject.scenarios(optional = True)
baseScenarioTable = allScenarios[allScenarios.ScenarioId == baselineID]
altrScenarioTable = allScenarios[allScenarios.ScenarioId == alternativeID]

# Load Results Scenario for the baseline scenario
if "Yes" in np.unique(baseScenarioTable.IsResult):
    baseScenario = myLibrary.scenarios(baselineID)
else:
    baseScenarios = allScenarios[allScenarios.ParentId == baselineID]
    if baseScenarios.empty:
        sys.exit("No results were found for the Baseline Scenario.")
    else:
        baseResultID = max(baseScenarios.ScenarioId)
        baseScenario = myLibrary.scenarios(baseResultID)

# Load Results Scenario for the alternative scenario
if "Yes" in np.unique(altrScenarioTable.IsResult):
    altrScenario = myLibrary.scenarios(alternativeID)
else:
    altrScenarios = allScenarios[allScenarios.ParentId == alternativeID]
    if altrScenarios.empty:
        sys.exit("No results were found for the Alternative Scenario.")
    else:
        altrResultID = max(altrScenarios.ScenarioId)
        altrScenario = myLibrary.scenarios(altrResultID)

# Load input datasheets for each scenario
baseOmniscapeOutput = baseScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
altrOmniscapeOutput = altrScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
baseEnsembleOutput = baseScenario.datasheets(name = "omniscape_outputSpatialEnsemble", show_full_paths = True)
altrEnsembleOutput = altrScenario.datasheets(name = "omniscape_outputSpatialEnsemble", show_full_paths = True)
baseRasterPath = baseScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
altrRasterPath = altrScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
baseTabular = baseScenario.datasheets(name = "omniscape_outputTabularReclassification")
altrTabular = altrScenario.datasheets(name = "omniscape_outputTabularReclassification")
baseThresholds = baseScenario.datasheets(name = "omniscape_reclassificationThresholds")
altrThresholds = altrScenario.datasheets(name = "omniscape_reclassificationThresholds")



# Choose the connectivity surface to compare -----------------------------------

# A Scenario carries either a single-model 'Normalized current' raster from
# omniscape's 'Omniscape' transformer or an 'Ensemble connectivity' raster from
# its 'Ensemble Connectivity' transformer, which combines several Scenarios
# (typically one per species). Both are continuous connectivity surfaces on the
# same grid, so an impact assessment reads whichever one each Scenario has -
# provided both have the same one, since the two are not interchangeable.

baseSurfacePath, baseSurfaceKind, baseSurfaceLabel = resolveConnectivitySurface(
    baseOmniscapeOutput, baseEnsembleOutput, "Baseline")
altrSurfacePath, altrSurfaceKind, altrSurfaceLabel = resolveConnectivitySurface(
    altrOmniscapeOutput, altrEnsembleOutput, "Alternative")

validateComparableSurfaces(baseSurfaceKind, altrSurfaceKind,
                           baseSurfaceLabel, altrSurfaceLabel)

surfaceKind = baseSurfaceKind
surfaceLabel = baseSurfaceLabel

safeUpdateRunLog("Comparing '" + surfaceLabel + "' rasters: Baseline "
                 + os.path.basename(baseSurfacePath) + ", Alternative "
                 + os.path.basename(altrSurfacePath) + ".")



# Validation for baseline & alternative scenarios results ----------------------

if (baseRasterPath.empty) | (altrRasterPath.empty):
    if (baseRasterPath.empty) & (altrRasterPath.empty):
        safeUpdateRunLog("'Connectivity categories' raster files are missing. Therefore, only the '" + surfaceLabel + "' raster files were used.")
    else:
        safeUpdateRunLog("The 'Connectivity categories' raster for one of the Scenarios was missing. Therefore, only the '" + surfaceLabel + "' raster files were used.")

if (baseTabular.empty) | (altrTabular.empty):
    if (baseTabular.empty) & (altrTabular.empty):
        safeUpdateRunLog("'Connectivity Categories Summary' datasheets are missing. Therefore, no tabular summary was calculated.") 
    else:
        safeUpdateRunLog("The 'Connectivity Categories Summary' datasheet for one of the Scenarios was missing. Therefore, no tabular summary was calculated.")



# Load rasters & validate that the two Scenarios are comparable -----------------

# Every output of this package is a pixel-by-pixel comparison between the two
# Scenarios, so the rasters must describe the same grid. Their valid areas are
# allowed to differ: comparisons are made over the pixels valid in BOTH
# Scenarios (the union of the two no-data masks), and any pixel valid in only
# one is excluded and reported to the run log. Both raster pairs are loaded and
# checked here, before anything is written, so that a failed grid check cannot
# leave a partial set of results behind.

hasCategories = (len(baseRasterPath) != 0) & (len(altrRasterPath) != 0)

# Connectivity categories are only comparable if both Scenarios cut the surface
# up at the same places. If they did not, the category rasters describe
# different things, so every category-derived output is skipped. The continuous
# surface comparison is unaffected and still runs.
if hasCategories:
    categoriesComparable, categoriesReason = categoriesAreComparable(
        baseTabular, altrTabular, baseThresholds, altrThresholds)

    if not categoriesComparable:
        hasCategories = False
        safeUpdateRunLog(
            categoriesReason + " All connectivity category outputs have been "
            "skipped. Only the '" + surfaceLabel + "' comparison was calculated.")

# Load the continuous connectivity surfaces
baseNormRaster = rasterio.open(baseSurfacePath)
altrNormRaster = rasterio.open(altrSurfacePath)
validateSameGrid(baseNormRaster, altrNormRaster, surfaceLabel)
# Read as float so that the subtraction below cannot wrap around
baseNormData = baseNormRaster.read().astype(float)
altrNormData = altrNormRaster.read().astype(float)
# Identify no-data pixels from the raster itself rather than assuming -9999
baseNormMask = nodataMask(baseNormRaster, baseNormData)
altrNormMask = nodataMask(altrNormRaster, altrNormData)
footprintDisagreement = reportNodataFootprint(baseNormMask, altrNormMask, surfaceLabel)
normMask = baseNormMask | altrNormMask

if hasCategories:
    # Load connectivity category rasters
    baseRaster = rasterio.open(baseRasterPath.movementTypes[0])
    altrRaster = rasterio.open(altrRasterPath.movementTypes[0])
    validateSameGrid(baseRaster, altrRaster, "Connectivity categories")
    baseData = baseRaster.read()
    altrData = altrRaster.read()
    # Identify no-data pixels from the raster itself rather than assuming -9999
    baseCategoryMask = nodataMask(baseRaster, baseData)
    altrCategoryMask = nodataMask(altrRaster, altrData)
    footprintDisagreement += reportNodataFootprint(baseCategoryMask, altrCategoryMask, "Connectivity categories")
    categoryMask = baseCategoryMask | altrCategoryMask



# Calculate spatial differences & Jaccard similarity ---------------------------

safeProgressBar(message="Calculating spatial differences", report_type="message")

# Continuous connectivity surface ----------------

# Each kind of surface gets its own file name and its own output column, so
# that the map legend names the quantity actually being shown and an ensemble
# comparison is never mistaken for a single-model one.
if surfaceKind == "ensemble":
    surfaceFileName = "ensembleConnectivityImpact.tif"
    surfaceOutputColumn = "overallEnsembleDifferenceRaster"
else:
    surfaceFileName = "normalizedCurrentImpact.tif"
    surfaceOutputColumn = "overallCurrentDifferenceRaster"

# Neutralise no-data pixels before the subtraction so that they cannot
# contribute an extreme value or propagate NaN into the result
baseNormClean = np.where(normMask, 0.0, baseNormData)
altrNormClean = np.where(normMask, 0.0, altrNormData)
# Calculate the overall impact of the intervention as absolute change
normDifference = altrNormClean - baseNormClean
# Set NA back to -9999
normDifference[normMask] = NODATA_VALUE
# Save output raster to file, declaring the no-data value explicitly so that
# it is not inherited from the input raster (which may not declare one)
outMeta = baseNormRaster.meta.copy()
outMeta.update(dtype = "float32", nodata = NODATA_VALUE)
surfaceOutputPath = os.path.join(outputOverallPath, surfaceFileName)
with rasterio.open(surfaceOutputPath, mode="w", **outMeta) as outputRaster:
    outputRaster.write(normDifference.astype("float32"))
# Load empty output datasheet
outputSpatialOverall = myScenario.datasheets(name = "omniscapeImpact_outputSpatialOverall")
# Save path the to file
outputSpatialOverall[surfaceOutputColumn] = pd.Series(surfaceOutputPath)
# Save outputs to SyncroSim Library
myParentScenario.save_datasheet(name = "omniscapeImpact_outputSpatialOverall", data = outputSpatialOverall)


# Connectivity categories ------------------------

if hasCategories:
    # Neutralise no-data pixels before the subtraction so that they cannot
    # contribute an extreme value or overflow the integer type
    baseClean = np.where(categoryMask, 0, baseData).astype(np.int32)
    altrClean = np.where(categoryMask, 0, altrData).astype(np.int32)
    # Calculate the overall impact of the intervention
    overallImpact = altrClean - baseClean
    # Set NA back to -9999
    overallImpact[categoryMask] = NODATA_VALUE
    # Save output raster to file, declaring the no-data value explicitly so that
    # it is not inherited from the input raster (which may not declare one)
    outMeta = baseRaster.meta.copy()
    outMeta.update(dtype = "int16", nodata = NODATA_VALUE)
    with rasterio.open(
        os.path.join(outputOverallPath, "connectivityCategoryImpact.tif"),
        mode="w", **outMeta) as outputRaster:
        outputRaster.write(overallImpact.astype("int16"))
    # Save path the to file
    outputSpatialOverall.overallDifferenceRaster = pd.Series(os.path.join(outputOverallPath, "connectivityCategoryImpact.tif"))
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputSpatialOverall", data = outputSpatialOverall)
    # Jaccard dissimilarity --------------------------
    # Get unique connectivity categories, ignoring no-data pixels so that a
    # no-data value coinciding with a category ID cannot be picked up as a class
    unique = np.unique(baseData[~categoryMask])
    # Transform array into dataframe
    unique = pd.DataFrame(unique)
    # Remove NA value
    uniqueClass = unique[(unique[0].isin(movementTypeClasses.classID))]
    baseReclassList = []
    altrReclassList = []
    # Load empty output datasheet
    outputSpatialCategory = myScenario.datasheets(name = "omniscapeImpact_outputSpatialCategory")
    outputTabularJaccard = myScenario.datasheets(name = "omniscapeImpact_outputTabularJaccard")
    # For each connectivity category
    for i in uniqueClass[0]:
        # Create binary map, excluding no-data pixels so that they cannot be
        # counted as belonging to this category
        baseTempRaster = ((baseData == i) & ~categoryMask).astype(np.int16)
        altrTempRaster = ((altrData == i) & ~categoryMask).astype(np.int16)
        baseReclassList.append(baseTempRaster)
        altrReclassList.append(altrTempRaster)
        # Calculate the difference between alternative and baseline scenarios
        differenceRaster = altrTempRaster - baseTempRaster
        similarityRaster = altrTempRaster + baseTempRaster
        # Set 0 to NA using -9999 flag. Note that this deliberately does not
        # distinguish real no-data from "this category is absent in both
        # Scenarios" - both are flagged -9999, as in previous versions.
        differenceRaster[(differenceRaster == 0) & (similarityRaster != 2)] = NODATA_VALUE
        # Save output raster to file
        with rasterio.open(os.path.join(outputCategoryPath, "connectivityDifference_" + repr(i) + ".tif"), mode="w", **outMeta) as outputRaster: outputRaster.write(differenceRaster)
        # Get internal ID for the connectivity category
        movementTypeID = movementTypeClasses.movementTypesId[movementTypeClasses.classID == i]
        # Save path the to file
        outputSpatialCategory.loc[len(outputSpatialCategory.index)] = [int(movementTypeID), os.path.join(outputCategoryPath, "connectivityDifference_" + repr(i) + ".tif")] 
        # Calculate Jaccard similarity
        rasterIntersection = similarityRaster == 2
        rasterUnion = similarityRaster >= 1
        jaccardDissimilarity = 1 - (rasterIntersection.sum() / rasterUnion.sum())
        # Save values
        outputTabularJaccard.loc[len(outputTabularJaccard.index)] = [int(movementTypeID), jaccardDissimilarity] 
    # Change movementTypeID from float to integer
    outputTabularJaccard.movementTypesID = outputTabularJaccard.movementTypesID.astype(int)
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputSpatialCategory", data = outputSpatialCategory)
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularJaccard", data = outputTabularJaccard)



# Calculate tabular differences ------------------------------------------------

safeProgressBar(message="Calculating tabular differences", report_type="message")

if (len(baseTabular) != 0) & (len(altrTabular) != 0):
    # Calculate change in area and percent cover. The two summaries are joined
    # on connectivity category rather than subtracted by row position, because
    # omniscape omits any category that occupies no pixels - so the two
    # Scenarios can hold different categories in different orders.
    diffSummary = alignCategorySummaries(baseTabular, altrTabular)
    # Change movementTypesID from string to class, then save. The differences
    # summary is derived from the tabular datasheets alone, so it is produced
    # even when the connectivity categories themselves are not comparable.
    movementStringToClass = pd.DataFrame({'movementTypesID': movementTypeClasses.movementTypesId,
                                        'Name': movementTypeClasses.Name})
    dS2C = movementStringToClass.set_index('Name').to_dict()
    diffSummary = diffSummary.replace(dS2C['movementTypesID'])
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularDifferences", data = diffSummary)

    # The area and proportion figures in these tables come from omniscape and
    # are computed over each Scenario's own valid area. When the two valid
    # areas differ, the figures being subtracted have different denominators,
    # unlike the raster-derived outputs, which compare only the shared area.
    if footprintDisagreement > 0:
        safeUpdateRunLog(
            "NOTE: The Scenarios' valid areas differ, so the 'Differences "
            "Summary' compares areas and proportions computed over each "
            "Scenario's own extent, not the shared extent.")


# The transitions summary is counted from the connectivity category rasters, so
# unlike the differences summary above it can only be produced when those
# rasters exist and are comparable between the two Scenarios.

if (len(baseTabular) != 0) & (len(altrTabular) != 0) & hasCategories:
    # Get unique connectivity categories
    uniqueCategory = pd.unique(movementTypeClasses['classID'].astype('int16'))
    # Get list of all possible combinations of change between connectivity categories
    categoryTransitions = list(itertools.product(uniqueCategory, uniqueCategory))
    # Load empty output datasheet
    outputTabularChange = myScenario.datasheets("omniscapeImpact_outputTabularChange")
    # For each connectivity category
    for transition in categoryTransitions:
        # Create a binary map for a category and scenario, excluding no-data
        # pixels so that they cannot be counted as belonging to a category
        baseClassRaster = ((baseData == transition[0]) & ~categoryMask) * 1
        altrClassRaster = ((altrData == transition[1]) & ~categoryMask) * 1
        # Calculate binary map sum
        sumRaster = baseClassRaster + altrClassRaster
        # Identify where category X transitioned to category Y
        transitionRaster = (sumRaster == 2) * 1
        unique, counts = np.unique(transitionRaster, return_counts = True)
        unique = pd.DataFrame(unique)
        unique[0] = unique[0].astype(int)
        freq = pd.DataFrame(counts)
        uniqueFreq = pd.concat([unique, freq], axis = 1, ignore_index = True)
        if 1 in uniqueFreq[0]:
            percentCover = uniqueFreq[1]/uniqueFreq[1].sum() 
            amountArea = (uniqueFreq[1] * baseRaster.res[1] * baseRaster.res[1])/1000000
            tempTabularChange = pd.concat([uniqueFreq[0], amountArea, percentCover], axis = 1, ignore_index = True)
            tempTabularChange = tempTabularChange[tempTabularChange[0] == 1]
            outputTabularChange.loc[len(outputTabularChange.index)] = [int(transition[0]), int(transition[1]), float(tempTabularChange[1]), float(tempTabularChange[2])]
        else:
            percentCover = 0
            amountArea = 0
            outputTabularChange.loc[len(outputTabularChange.index)] = [int(transition[0]), int(transition[1]), float(amountArea), float(percentCover)]
    # Change movementTypesID class to string
    movementClassToString = pd.DataFrame({'classID': movementTypeClasses.classID.astype(float),
                                        'Name': movementTypeClasses.Name})
    dC2S = movementClassToString.set_index('classID').to_dict()
    outputTabularChange = outputTabularChange.replace(dC2S['Name'])
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularChange", data = outputTabularChange)


