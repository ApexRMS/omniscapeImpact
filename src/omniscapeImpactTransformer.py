## omniscapeImpact

import pysyncrosim as ps
import pandas as pd
import os
import rasterio
import numpy as np
import itertools
import sys

# Validation for base package version ------------------------------------------

mySession = ps.Session() 
packagesInstalled = mySession.packages()
omniscapeVersion = packagesInstalled.Version[packagesInstalled.Name == "omniscape"]

if pd.unique(omniscapeVersion)[0] != "1.1.1":
    sys.exit("The omniscapeImpact add-on package version 1.0.1 requires the omniscape base package version 1.1.1.")



# Set up -----------------------------------------------------------------------

ps.environment.progress_bar(message="Setting up Scenario", report_type="message")

# Set environment and working directory
e = ps.environment._environment()
wrkDir = e.output_directory.item()

# Open SyncroSim Library, Project and Scenario
myLibrary = ps.Library()
myProject = myLibrary.projects(pid = 1) 
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)
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
baseScenarioTable = allScenarios[allScenarios.ScenarioID == int(differenceScenarios.Baseline[0])]
altrScenarioTable = allScenarios[allScenarios.ScenarioID == int(differenceScenarios.Alternative[0])]

# Load Results Scenario for the baseline scenario
if "Yes" in np.unique(baseScenarioTable.IsResult):
    baseScenario = myLibrary.scenarios(int(differenceScenarios.Baseline[0]))
else:
    baseScenarios = allScenarios[allScenarios.ParentID == int(differenceScenarios.Baseline[0])]
    if baseScenarios.empty:
        sys.exit("No results were found for the Baseline Scenario.")
    else:
        baseResultID = max(baseScenarios.ScenarioID)
        baseScenario = myLibrary.scenarios(baseResultID)

# Load Results Scenario for the alternative scenario
if "Yes" in np.unique(altrScenarioTable.IsResult):
    altrScenario = myLibrary.scenarios(int(differenceScenarios.Alternative[0]))
else:
    altrScenarios = allScenarios[allScenarios.ParentID == int(differenceScenarios.Alternative[0])]
    if altrScenarios.empty:
        sys.exit("No results were found for the Alternative Scenario.")
    else:
        altrResultID = max(altrScenarios.ScenarioID)
        altrScenario = myLibrary.scenarios(altrResultID)

# Load input datasheets for each scenario
baseOmniscapeOutput = baseScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
altrOmniscapeOutput = altrScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
baseRasterPath = baseScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
altrRasterPath = altrScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)
baseTabular = baseScenario.datasheets(name = "omniscape_outputTabularReclassification")
altrTabular = altrScenario.datasheets(name = "omniscape_outputTabularReclassification")



# Validation for baseline & alternative scenarios results ----------------------

if baseOmniscapeOutput.normalized_cum_currmap[0] != baseOmniscapeOutput.normalized_cum_currmap[0]:
    sys.exit("'Normalized current' raster is required for the Baseline Scenario.")

if altrOmniscapeOutput.normalized_cum_currmap[0] != altrOmniscapeOutput.normalized_cum_currmap[0]:
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



# Calculate spatial differences & Jaccard similarity ---------------------------

ps.environment.progress_bar(message="Calculating spatial differences", report_type="message")

# Normalized current -----------------------------

if (baseOmniscapeOutput.normalized_cum_currmap[0] == baseOmniscapeOutput.normalized_cum_currmap[0]) & (altrOmniscapeOutput.normalized_cum_currmap[0] == altrOmniscapeOutput.normalized_cum_currmap[0]):
    # Load normalized current raster
    baseNormRaster = rasterio.open(baseOmniscapeOutput.normalized_cum_currmap[0])
    altrNormRaster = rasterio.open(altrOmniscapeOutput.normalized_cum_currmap[0])
    # Transform raster into dataframe
    baseNormData = baseNormRaster.read()
    altrNormData = altrNormRaster.read()
    # Reset data as float
    baseNormData.astype(float) 
    altrNormData.astype(float) 
    # Calculate the overall impact of the intervention as absolute change
    normDifference = altrNormData - baseNormData
    # Set NA back to -9999
    normDifference[(baseNormData == -9999) & (altrNormData == -9999)] = -9999
    # Save output raster to file
    outMeta = baseNormRaster.meta
    with rasterio.open(
        os.path.join(outputOverallPath, "normalizedCurrentImpact.tif"), 
        mode="w", **outMeta) as outputRaster:
        outputRaster.write(normDifference)
    # Load empty output datasheet
    outputSpatialOverall = myScenario.datasheets(name = "omniscapeImpact_outputSpatialOverall")
    # Save path the to file
    outputSpatialOverall.overallCurrentDifferenceRaster = pd.Series(os.path.join(outputOverallPath, "normalizedCurrentImpact.tif"))
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputSpatialOverall", data = outputSpatialOverall)


# Connectivity categories ------------------------

if (len(baseRasterPath) != 0) & (len(altrRasterPath) != 0):
    # Load connectivity category raster
    baseRaster = rasterio.open(baseRasterPath.movement_types[0])
    altrRaster = rasterio.open(altrRasterPath.movement_types[0])
    # Transform raster into dataframe
    baseData = baseRaster.read()
    altrData = altrRaster.read()
    baseData.astype(float) 
    altrData = altrRaster.read()
    # Calculate the overall impact of the intervention
    overallImpact = altrData - baseData
    # Set NA back to -9999
    overallImpact[(altrData == -9999) & (baseData == -9999)] = -9999
    # Save output raster to file
    outMeta = baseRaster.meta
    with rasterio.open(
        os.path.join(outputOverallPath, "connectivityCategoryImpact.tif"), 
        mode="w", **outMeta) as outputRaster:
        outputRaster.write(overallImpact)
    # Save path the to file
    outputSpatialOverall.overallDifferenceRaster = pd.Series(os.path.join(outputOverallPath, "connectivityCategoryImpact.tif"))
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputSpatialOverall", data = outputSpatialOverall)
    # Jaccard dissimilarity --------------------------
    # Get unique connectivity categories
    unique = np.unique(baseData)
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
        # Create a copy of the connectivity category dataframe
        baseTempRaster = baseData.copy()
        altrTempRaster = altrData.copy()
        # Create binary map
        baseTempRaster[np.where(baseData != i)] = 0 
        baseTempRaster[np.where(baseData == i)] = 1
        altrTempRaster[np.where(altrData != i)] = 0
        altrTempRaster[np.where(altrData == i)] = 1
        baseReclassList.append(baseTempRaster)
        altrReclassList.append(altrTempRaster)
        # Calculate the difference between alternative and baseline scenarios
        differenceRaster = altrTempRaster - baseTempRaster
        similarityRaster = altrTempRaster + baseTempRaster
        # Set 0 to NA using -9999 flag
        differenceRaster[(differenceRaster == 0) & (similarityRaster != 2)] = -9999
        # Save output raster to file
        with rasterio.open(os.path.join(outputCategoryPath, "connectivityDifference_" + repr(i) + ".tif"), mode="w", **outMeta) as outputRaster: outputRaster.write(differenceRaster)
        # Get internal ID for the connectivity category
        movementTypeID = movementTypeClasses.movementTypesID[movementTypeClasses.classID == i]
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
    # Get unique connectivity categories
    uniqueCategory = pd.unique(movementTypeClasses['classID'].astype('int16'))
    # Get list of all possible combinations of change between connectivity categories
    categoryTransitions = list(itertools.product(uniqueCategory, uniqueCategory))
    # Load empty output datasheet
    outputTabularChange = myScenario.datasheets("omniscapeImpact_outputTabularChange")
    # For each connectivity category
    for transition in categoryTransitions:
        # Create a copy of the connectivity category dataframe
        baseTempRaster = baseData.copy()
        altrTempRaster = altrData.copy()
        # Create a binary map for a category and scenario
        baseClassRaster = (baseTempRaster == transition[0]) * 1
        altrClassRaster = (altrTempRaster == transition[1]) * 1
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
    # Change movementTypesID string to class
    movementStringToClass = pd.DataFrame({'movementTypesID': movementTypeClasses.movementTypesID,
                                        'Name': movementTypeClasses.Name})
    dS2C = movementStringToClass.set_index('Name').to_dict()
    diffSummary = diffSummary.replace(dS2C['movementTypesID'])
    # Change movementTypesID class to string
    movementClassToString = pd.DataFrame({'classID': movementTypeClasses.classID.astype(float),
                                        'Name': movementTypeClasses.Name})
    dC2S = movementClassToString.set_index('classID').to_dict()
    outputTabularChange = outputTabularChange.replace(dC2S['Name'])
    # Save outputs to SyncroSim Library
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularDifferences", data = diffSummary)
    myParentScenario.save_datasheet(name = "omniscapeImpact_outputTabularChange", data = outputTabularChange)


