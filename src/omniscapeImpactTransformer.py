## omniscapeImpact

import pysyncrosim as ps
import pandas as pd
import os
import rasterio
import numpy as np
import itertools
import sys

from helperFunctions import validateNodataFootprint, validateSameGrid, nodataMask, sameCategoryThresholds
from constants import NODATA_VALUE

# Validation for base package version ------------------------------------------

mySession = ps.Session() 
packagesInstalled = mySession.packages()
omniscapeVersion = packagesInstalled.Version[packagesInstalled.Name == "omniscape"]


# Set up -----------------------------------------------------------------------

ps.environment.progress_bar(message="Setting up Scenario", report_type="message")

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
differenceScenarios = myScenario.datasheets(name = "omniscapeImpact_differenceScenarios")



# Validation for inputs --------------------------------------------------------

if movementTypeClasses.empty:
    sys.exit("'Category Thresholds' are required.")

if (len(differenceScenarios.Baseline) == 0) | (len(differenceScenarios.Alternative) == 0):
    sys.exit("'Baseline Scenario ID' and 'Alternative Scenario ID' are required.")



# Open baseline and alternative scenario results -------------------------------

# Detect if Parent or Result Scenario IDs
allScenarios = myProject.scenarios(optional = True)
baseScenarioTable = allScenarios[allScenarios.ScenarioId == int(differenceScenarios.Baseline[0])]
altrScenarioTable = allScenarios[allScenarios.ScenarioId == int(differenceScenarios.Alternative[0])]

# Load Results Scenario for the baseline scenario
if "Yes" in np.unique(baseScenarioTable.IsResult):
    baseScenario = myLibrary.scenarios(int(differenceScenarios.Baseline[0]))
else:
    baseScenarios = allScenarios[allScenarios.ParentId == int(differenceScenarios.Baseline[0])]
    if baseScenarios.empty:
        sys.exit("No results were found for the Baseline Scenario.")
    else:
        baseResultID = max(baseScenarios.ScenarioId)
        baseScenario = myLibrary.scenarios(baseResultID)

# Load Results Scenario for the alternative scenario
if "Yes" in np.unique(altrScenarioTable.IsResult):
    altrScenario = myLibrary.scenarios(int(differenceScenarios.Alternative[0]))
else:
    altrScenarios = allScenarios[allScenarios.ParentId == int(differenceScenarios.Alternative[0])]
    if altrScenarios.empty:
        sys.exit("No results were found for the Alternative Scenario.")
    else:
        altrResultID = max(altrScenarios.ScenarioId)
        altrScenario = myLibrary.scenarios(altrResultID)

# Load input datasheets for each scenario
baseOmniscapeOutput = baseScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
altrOmniscapeOutput = altrScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
baseRasterPath = baseScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
altrRasterPath = altrScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
baseTabular = baseScenario.datasheets(name = "omniscape_outputTabularReclassification")
altrTabular = altrScenario.datasheets(name = "omniscape_outputTabularReclassification")
baseThresholds = baseScenario.datasheets(name = "omniscape_reclassificationThresholds")
altrThresholds = altrScenario.datasheets(name = "omniscape_reclassificationThresholds")



# Validation for baseline & alternative scenarios results ----------------------

if baseOmniscapeOutput.normalizedCumCurrmap[0] != baseOmniscapeOutput.normalizedCumCurrmap[0]:
    sys.exit("'Normalized current' raster is required for the Baseline Scenario.")

if altrOmniscapeOutput.normalizedCumCurrmap[0] != altrOmniscapeOutput.normalizedCumCurrmap[0]:
    sys.exit("'Normalized current' raster is required for the Alternative Scenario.")

if (baseRasterPath.empty) | (altrRasterPath.empty):
    if (baseRasterPath.empty) & (altrRasterPath.empty):
        ps.environment.update_run_log("'Connectivity categories' raster files are missing. Therefore, only the 'Normalized current' raster files were used.") 
    else:
        ps.environment.update_run_log("The 'Connectivity categories' raster for one of the Scenarios was missing. Therefore, only the 'Normalized current' raster files were used.") 

if (baseTabular.empty) | (altrTabular.empty):
    if (baseTabular.empty) & (altrTabular.empty):
        ps.environment.update_run_log("'Connectivity Categories Summary' datasheets are missing. Therefore, no tabular summary was calculated.") 
    else:
        ps.environment.update_run_log("The 'Connectivity Categories Summary' datasheet for one of the Scenarios was missing. Therefore, no tabular summary was calculated.")



# Load rasters & validate that the two Scenarios are comparable -----------------

# Every output of this package is a pixel-by-pixel comparison between the two
# Scenarios, so the rasters must describe the same grid and must agree about
# which pixels hold valid data. Both raster pairs are loaded and checked here,
# before anything is written, so that a failed check cannot leave a partial set
# of results behind.

hasNormalizedCurrent = ((baseOmniscapeOutput.normalizedCumCurrmap[0] == baseOmniscapeOutput.normalizedCumCurrmap[0])
                        & (altrOmniscapeOutput.normalizedCumCurrmap[0] == altrOmniscapeOutput.normalizedCumCurrmap[0]))
hasCategories = (len(baseRasterPath) != 0) & (len(altrRasterPath) != 0)

# Connectivity categories are only comparable if both Scenarios were classified
# using the same thresholds. If they were not, the category rasters describe
# different things, so every category-derived output is skipped. The
# 'Normalized current' comparison is unaffected and still runs.
if hasCategories and not sameCategoryThresholds(baseThresholds, altrThresholds):
    hasCategories = False
    ps.environment.update_run_log(
        "The Baseline and Alternative Scenarios use different 'Category Thresholds'. "
        "Connectivity categories are therefore not comparable between them, and all "
        "connectivity category outputs have been skipped. Only the 'Normalized "
        "current' comparison was calculated.")

if hasNormalizedCurrent:
    # Load normalized current rasters
    baseNormRaster = rasterio.open(baseOmniscapeOutput.normalizedCumCurrmap[0])
    altrNormRaster = rasterio.open(altrOmniscapeOutput.normalizedCumCurrmap[0])
    validateSameGrid(baseNormRaster, altrNormRaster, "Normalized current")
    # Read as float so that the subtraction below cannot wrap around
    baseNormData = baseNormRaster.read().astype(float)
    altrNormData = altrNormRaster.read().astype(float)
    # Identify no-data pixels from the raster itself rather than assuming -9999
    baseNormMask = nodataMask(baseNormRaster, baseNormData)
    altrNormMask = nodataMask(altrNormRaster, altrNormData)
    validateNodataFootprint(baseNormMask, altrNormMask, "Normalized current")
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
    validateNodataFootprint(baseCategoryMask, altrCategoryMask, "Connectivity categories")
    categoryMask = baseCategoryMask | altrCategoryMask



# Calculate spatial differences & Jaccard similarity ---------------------------

ps.environment.progress_bar(message="Calculating spatial differences", report_type="message")

# Normalized current -----------------------------

if hasNormalizedCurrent:
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
    with rasterio.open(
        os.path.join(outputOverallPath, "normalizedCurrentImpact.tif"),
        mode="w", **outMeta) as outputRaster:
        outputRaster.write(normDifference.astype("float32"))
    # Load empty output datasheet
    outputSpatialOverall = myScenario.datasheets(name = "omniscapeImpact_outputSpatialOverall")
    # Save path the to file
    outputSpatialOverall.overallCurrentDifferenceRaster = pd.Series(os.path.join(outputOverallPath, "normalizedCurrentImpact.tif"))
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

ps.environment.progress_bar(message="Calculating tabular differences", report_type="message")

if (len(baseTabular) != 0) & (len(altrTabular) != 0):
    # Calculate change in area and percent cover
    diffArea = altrTabular.amountArea - baseTabular.amountArea  
    diffCover = altrTabular.percentCover - baseTabular.percentCover
    # Create tabular output
    diffSummary = pd.concat([baseTabular.movementTypesID, diffArea, diffCover], axis = 1, ignore_index = True)
    diffSummary = diffSummary.rename(columns = {0: "movementTypesID", 1:"amountAreaDifference", 2:"percentCoverDifference"})
    # Change movementTypesID from string to class, then save. The differences
    # summary is derived from the tabular datasheets alone, so it is produced
    # even when the connectivity categories themselves are not comparable.
    movementStringToClass = pd.DataFrame({'movementTypesID': movementTypeClasses.movementTypesId,
                                        'Name': movementTypeClasses.Name})
    dS2C = movementStringToClass.set_index('Name').to_dict()
    diffSummary = diffSummary.replace(dS2C['movementTypesID'])
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularDifferences", data = diffSummary)


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


