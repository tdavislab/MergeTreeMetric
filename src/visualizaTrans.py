"""Interactive Visualization Tools for A Structural Average of Labeled Merge Trees"""
# Author: Lin Yan <lynne.h.yan@gmail.com>
import os
import sys
import re
import numpy as np
import vtk

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

def createVisFile(p, outFile):
    points = vtk.vtkPoints()
    points.InsertPoint(0, p[0:3])
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    scalars_ = vtk.vtkFloatArray()
    scalars_.SetNumberOfValues(1)
    scalars_.SetValue(0, p[3])

    polydata.GetPointData().SetScalars(scalars_)

    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()

        
    writer = vtk.vtkXMLPolyDataWriter();
    writer.SetFileName(outFile);
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
    
    
def main():
    fileDir = sys.argv[1]
    scalarField =  sys.argv[2]
    IL =  sys.argv[3]
    extending_mode = sys.argv[4]
    PG = sys.argv[5]
    flag = int(sys.argv[6])

    outDir =  os.path.join(fileDir, 'Output')
    diagDir = os.path.join(outDir, 'Diagnose')
    fileInfo = "_" +scalarField + "_" + PG + "_" + IL + "_" + extending_mode
    transFiles = os.path.join(diagDir, 'transCritical' + fileInfo + '.txt')
    transBackFiles = os.path.join(diagDir, 'transCriticalBack' + fileInfo + '.txt')

    transCP = np.loadtxt(transFiles)
    transBackCP = np.loadtxt(transBackFiles)
    inputFiles = getInputFiles(fileDir)

    visDir =  os.path.join(diagDir, 'visTrans_'+scalarField+"_"+ IL + "_" + extending_mode)
    make_dir(visDir)
    
    for idx in range(len(inputFiles)):
        visFileName = "visTrans_" + str(int(re.search(r'\d+', inputFiles[idx]).group())) + '.vtp'
        visFile = os.path.join(visDir, visFileName)
        createVisFile(transCP[idx], visFile)
        visBackFileName = "visBackTrans_" + str(int(re.search(r'\d+', inputFiles[idx]).group())) + '.vtp'
        visBackFile = os.path.join(visDir, visBackFileName)
        createVisFile(transBackCP[idx], visBackFile)

    

if __name__ == '__main__':
    main()
    
