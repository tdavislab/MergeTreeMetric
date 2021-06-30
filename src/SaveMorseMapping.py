import sys
import os
import re

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

import math


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def getInputFiles(fileDir):
    files = os.listdir(fileDir)
    fileList = []
    for i in files:
        if i.split(".")[-1]=="vtp" or i.split(".")[-1]=="vti":
            fileList.append(i)
    order = []
    for i in range(len(fileList)):
        order.append(int(re.search(r'\d+', fileList[i]).group()))
    fileList = [x for _,x in sorted(zip(order,fileList))]
    return fileList

def getNcnt(treeFile):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(treeFile)
    reader.Update()

    Tree = reader.GetOutput()
    return int(Tree.GetNumberOfPoints())

def negativeScalarField(inputFile, outFile, sf):
    fileType = inputFile.split('.')[-1]
    outFile = outFile + fileType
    
    if fileType == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
        negData = vtk.vtkPolyData()
        writer = vtk.vtkXMLPolyDataWriter()
    if fileType == "vti":
        reader = vtk.vtkXMLImageDataReader()
        negData = vtk.vtkImageData()
        writer = vtk.vtkXMLImageDataWriter()
    reader.SetFileName(inputFile) 
    reader.Update()

    negData.CopyStructure(reader.GetOutput())
    scalar_ = vtk.util.numpy_support.numpy_to_vtk(0 - vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(sf)))
    scalar_.SetName(sf)
    negData.GetPointData().AddArray(scalar_)

    
    writer.SetInputDataObject(negData)
    writer.SetFileName(outFile)
    writer.Update()


def MorseMapping(dataDir, fileDir, file1, file2, PersThr, sf):
    submitCommand = "pvpython ./src/MorseMapping.py " + os.path.join(dataDir, file1)+ ' ' + os.path.join(dataDir, file2) + ' ' + fileDir + '/label_' + str(int(re.search(r'\d+', file2).group())) + '.txt' + ' ' + PersThr + ' ' + sf
    os.system(submitCommand)

def MorseMappingST(file1, file2, fileDir, fileName, PersThr, sf):
    submitCommand = "pvpython ./src/MorseMapping.py " + file1 + ' ' + file2 + ' ' + fileDir + '/label_' +  str(int(re.search(r'\d+', fileName).group())) + '.txt' + ' ' + PersThr + ' ' + sf
    os.system(submitCommand)
    
def main():
    fileDir = sys.argv[1]
    PersThr = sys.argv[2]
    treeType = sys.argv[3]
    scalarField = sys.argv[4]
    make_dir("./tmp/")

    fileType = 'vtp'
    inputFiles = getInputFiles(fileDir)

    interDir = os.path.join(fileDir, 'IntermediateFiles')
    mt_dir = os.path.join(interDir, 'MergeTrees')
    mtDir = os.path.join(mt_dir, treeType)
    ncnts = []
    for j in range(len(inputFiles)):
        treeFile = mtDir + '/edges_monoMesh_' + str(int(re.search(r'\d+', inputFiles[j]).group())) + '.vtk'
        ncnts.append(getNcnt(treeFile))
    pivot = np.argmax(ncnts)
    if treeType == 'st':
        fileType = inputFiles[pivot].split('.')[-1]
        inFileP = os.path.join(fileDir, inputFiles[pivot])
        outFileP = './tmp/negMeshP.'
        negativeScalarField(inFileP,  outFileP, scalarField)

    mpDir = os.path.join(interDir, 'MorseMappingLabels')
    mpDir = os.path.join(mpDir, treeType)
    for i in range(0, len(inputFiles)):
        if i != pivot:
            if treeType =='st':
                inFilei =  os.path.join(fileDir,  inputFiles[i])
                outFilei = './tmp/negMeshi.'
                negativeScalarField(inFilei,  outFilei, scalarField)
                MorseMappingST(outFileP + fileType, outFilei + fileType, mpDir, inputFiles[i], PersThr, scalarField)
            else:
                MorseMapping(fileDir, mpDir, inputFiles[pivot], inputFiles[i], PersThr, scalarField)    
               
        

if __name__ == '__main__':

    main()
  
  
   
