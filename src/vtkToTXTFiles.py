"""Interactive Visualization Tools for A Structural Average of Labeled Merge Trees"""
# Author: Lin Yan <lynne.h.yan@gmail.com>

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

def saveTreeFile(treeFile, nodesFile, edgesFile, treeType):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(treeFile)
    reader.Update()

    Tree = reader.GetOutput()
    nodes = Tree.GetPoints()
    scalar =  vtk_to_numpy(Tree.GetPointData().GetArray("Scalar"))
    if treeType == 'st':
        scalar = 1000 - scalar
    nodes = vtk_to_numpy(nodes.GetData())
    edges = Tree.GetCells()
    edges = vtk_to_numpy(edges.GetData())
    array_len = edges[0]
    links = []
    ecnt = int(len(edges)/(array_len+1))
    for j in range(0, ecnt):
        links.append([edges[j*(array_len+1)+1], edges[j*(array_len+1)+2]])
    links = np.array(links)
    nodes = np.concatenate((nodes, np.array([scalar]).T), axis=1)

    np.savetxt(nodesFile, nodes, fmt='%.6f')
    np.savetxt(edgesFile, links, fmt='%d')
    

def saveScalarFile(inputFileName, scalarFile, sf):
    fileType = inputFileName.split('.')[-1]
    reader = vtk.vtkXMLPolyDataReader()
    if fileType == "vti":
        reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(inputFileName) 
    reader.Update()
    polydata = reader.GetOutput()
    scalar = vtk_to_numpy(polydata.GetPointData().GetArray(sf))
    np.savetxt(scalarFile, scalar, fmt='%.6f')
    

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


def main():
    fileDir = sys.argv[1]
    treeType = sys.argv[2]
    scalarField = sys.argv[3]
    
    inputFiles = getInputFiles(fileDir)
    for inputFile in inputFiles:
        inputFileDir = os.path.join(fileDir, inputFile)
        interDir = os.path.join(fileDir, 'IntermediateFiles')
        mt_dir = os.path.join(interDir, 'MergeTrees')
        mtDir = os.path.join(mt_dir, treeType)
        sfDir = os.path.join(interDir, 'ScalarFields')
        scalarFile = os.path.join(sfDir, 'scalar_' +  str(int(re.search(r'\d+', inputFile).group()))+ '.txt')
        saveScalarFile(inputFileDir, scalarFile, scalarField)
        fileName = 'monoMesh_' + str(int(re.search(r'\d+', inputFile).group()))
        treeFile = os.path.join(mtDir, 'edges_' + fileName + '.vtk')
        txtDir = os.path.join(mtDir, 'TXTFormat')
        make_dir(txtDir)
        nodesFile = os.path.join(txtDir, 'treeNodes_' + fileName + '.txt')
        edgesFile = os.path.join(txtDir, 'treeEdges_' + fileName + '.txt')
        saveTreeFile(treeFile, nodesFile, edgesFile, treeType)
    

if __name__ == '__main__':

    main()
  
  
   
