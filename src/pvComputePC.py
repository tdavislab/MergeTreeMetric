#!/usr/bin/env pvpython

import sys
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
from paraview.simple import *

inputDir = sys.argv[1]
outfileName = sys.argv[2]
outputDir = sys.argv[3]
scalarField = sys.argv[4]
treeType = sys.argv[5]

inputFileName = inputDir
fileType = inputDir.split('.')[-1]

if fileType == 'vtp':
    inputData = XMLPolyDataReader(FileName=[inputFileName])
if fileType == 'vti':
    inputData = XMLImageDataReader(FileName=[inputFileName])
    
tetrahedralize = Tetrahedralize(Input=inputData)

# create a new 'TTK PersistenceCurve'
tTKPersistenceCurve = TTKPersistenceCurve(Input=tetrahedralize)
tTKPersistenceCurve.ScalarField = scalarField
tTKPersistenceCurve.InputOffsetField = scalarField

if treeType == 'st':
    SaveData(outputDir + outfileName, OutputPort(tTKPersistenceCurve, 2))
else:
    SaveData(outputDir + outfileName, OutputPort(tTKPersistenceCurve, 0))
