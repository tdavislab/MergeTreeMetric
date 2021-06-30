#!/usr/bin/env pvpython

import sys
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob
import os
from paraview.simple import *

inputDir = sys.argv[1]
fileName = sys.argv[2]
outputDir = sys.argv[3]
PersThr = float(sys.argv[4])
scalarField = sys.argv[5]
treeType = sys.argv[6]

inputFileName = inputDir
fileType = inputDir.split('.')[-1]
outputCritsFileName = outputDir + "crits_" + fileName + '.vtk'
outputEdgesFileName = outputDir + "edges_" + fileName + '.vtk'

if fileType == 'vtp':
    inputData = XMLPolyDataReader(FileName=[inputFileName])
if fileType == 'vti':
    inputData = XMLImageDataReader(FileName=[inputFileName])
    
processedData = Tetrahedralize(inputData)
persistenceDiagram = TTKPersistenceDiagram(processedData)
persistenceDiagram.ScalarField = scalarField



criticalPointPairs = Threshold(persistenceDiagram)
criticalPointPairs.Scalars = ['CELLS', 'PairIdentifier']
criticalPointPairs.ThresholdRange = [-0.1, 99999999]

persistentPairs = Threshold(criticalPointPairs)
persistentPairs.Scalars = ['CELLS', 'Persistence']
        
persistentPairs.ThresholdRange = [PersThr, 9999999999]
topologicalSimplification = TTKTopologicalSimplification(
    Domain=processedData, Constraints=persistentPairs, ScalarField = scalarField)



ftm = TTKMergeandContourTreeFTM(topologicalSimplification)
ftm.ScalarField = scalarField
if treeType == 'mt':
    ftm.TreeType = 0
else:
    ftm.TreeType = 'Split Tree'

SaveData(outputCritsFileName, OutputPort(ftm, 0), FileType='Ascii')
SaveData(outputEdgesFileName, OutputPort(ftm, 1), FileType='Ascii')
