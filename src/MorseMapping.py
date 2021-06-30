import sys
import math
import vtk
import os
import numpy as np
from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def distance(self, other):
        return self.euclidian_distance(other)
    
    def euclidian_distance(self, other):
        return math.sqrt((self.x - other.x) * (self.x - other.x)
            + (self.y - other.y) * (self.y - other.y)
            + (self.z - other.z) * (self.z - other.z))
    
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def __str__(self):
        return "(%f, %f, %f)" % (self.x, self.y, self.z)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

class CriticalPoint:
    def __init__(self, loc, vertexID, value, criticalType):
        self.loc = Point(loc[0], loc[1], loc[2])
        self.vertexID = vertexID
        self.value = value
        self.criticalType = criticalType
    
    def __str__(self):
        return "%s\n  loc: %s\n  vertexID: %d\n  value: %f\n  criticalType: %d" \
        % (type(self), self.loc.__str__(), self.vertexID, self.value, self.criticalType)
    
    def __eq__(self, other):
        return self.vertexID == other.vertexID 

def computeMergeTreeAndMSCSegmentation(inputFile, persistenceThreshold, sf):
	fileType = inputFile.split('.')[-1]
	if fileType == 'vtp':
		inputData = XMLPolyDataReader(FileName=[inputFile])
	if fileType == 'vti':
		inputData = XMLImageDataReader(FileName=[inputFile])
	processedData = Tetrahedralize(inputData)
	persistenceDiagram = TTKPersistenceDiagram(processedData)
	persistenceDiagram.ScalarField = sf

	pd = servermanager.Fetch(persistenceDiagram, idx=0)
	maxv = max(vtk_to_numpy(pd.GetCellData().GetArray("Persistence")))
        
	criticalPointPairs = Threshold(persistenceDiagram)
	criticalPointPairs.Scalars = ['CELLS', 'PairIdentifier']
	criticalPointPairs.ThresholdRange = [-0.1, 99999999]

	persistentPairs = Threshold(criticalPointPairs)
	persistentPairs.Scalars = ['CELLS', 'Persistence']
	print (persistenceThreshold)
	persistentPairs.ThresholdRange = [persistenceThreshold, maxv+1]
	topologicalSimplification = TTKTopologicalSimplification(Domain=processedData, Constraints=persistentPairs, ScalarField = sf)

	ftm = TTKMergeandContourTreeFTM(topologicalSimplification)
	ftm.ScalarField = sf
	ftm.TreeType = 0

	cps = servermanager.Fetch(ftm, idx=0)
	numPoints = cps.GetNumberOfPoints()
	minima = []
	for i in range(numPoints):
		loc = cps.GetPoint(i)
		vertexID = int(cps.GetPointData().GetArray("VertexId").GetTuple1(i))
		value = cps.GetPointData().GetArray("Scalar").GetTuple1(i)
		critType = int(cps.GetPointData().GetArray("CriticalType").GetTuple1(i))
		cp = CriticalPoint(loc, vertexID, value, critType)
		if critType == 0:
			minima.append(cp)

	morseSmaleComplex = TTKMorseSmaleComplex(topologicalSimplification)
	morseSmaleComplex.ScalarField = sf
	segmentation = servermanager.Fetch(morseSmaleComplex, idx=3)
	descSeg = segmentation.GetPointData().GetArray('DescendingManifold')

	return minima, descSeg

def getMapping(currMinima, nextMinima, nextSeg):
	mapping = {}
	for minima in currMinima:
		nextMin = int(nextSeg.GetTuple1(minima.vertexID))
		nextVertexID = nextMinima[nextMin].vertexID
		mapping[minima.vertexID] = nextVertexID
	return mapping

def getStrongAndWeakMappings(forwardMapping, backwardMapping):
	strongMappings = {}
	weakMappings = {}
	for currID in forwardMapping:
		nextID = forwardMapping[currID]
		backID = backwardMapping[nextID]
		if backID == currID:
			strongMappings[currID] = nextID
		else:
			weakMappings[currID] = nextID
	return strongMappings, weakMappings

def vId2nId(ID, l, loc):
    return loc[l.index(ID)]
   
def writeMappings(strongMappings, weakforwardMappings, weakbackwardMappings, outputFile, l1, loc1, l2, loc2):
	with open(outputFile, "w+") as file:
		file.write("Strong mappings:\n")
		for currID in strongMappings:
			pos1 = vId2nId(currID, l1, loc1)
			pos2 = vId2nId(strongMappings[currID], l2, loc2)                       
			file.write("%.2f %.2f %.2f %.2f %.2f %.2f\n" % (pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]))
		file.write("Weak forward mappings:\n")
		for currID in weakforwardMappings:
			pos1 = vId2nId(currID, l1, loc1)
			pos2 = vId2nId(weakforwardMappings[currID], l2, loc2)
			file.write("%.2f %.2f %.2f %.2f %.2f %.2f\n" % (pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]))
                                   
		file.write("Weak backward mappings:\n")
		for currID in weakbackwardMappings:
			pos1 = vId2nId(weakbackwardMappings[currID], l1, loc1)
			pos2 =vId2nId(currID, l2, loc2)
			file.write("%.2f %.2f %.2f %.2f %.2f %.2f\n" % (pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]))

def getMappedFromNodes(l1):
    vlist = []
    locs= []
    for i in range(len(l1)):
        vlist.append(l1[i].vertexID)
        locs.append([l1[i].loc.x, l1[i].loc.y, l1[i].loc.z])
    return vlist, locs

    
if __name__ == '__main__':
	if (len(sys.argv) < 5):
		print("Usage:\n\t pvpython MorseMapping.py <input file 1> <input file 2> <output maping file> <persistence threshold>")
		sys.exit(-1)

	inputFile1 = sys.argv[1]
	inputFile2 = sys.argv[2]
	outputFile = sys.argv[3]
	persThres = float(sys.argv[4])
	scalarField = sys.argv[5]

	minima1, seg1 = computeMergeTreeAndMSCSegmentation(inputFile1, persThres, scalarField)
	minima2, seg2 = computeMergeTreeAndMSCSegmentation(inputFile2, persThres, scalarField)
	l1, loc1 = getMappedFromNodes(minima1)
	l2, loc2 = getMappedFromNodes(minima2)
	
	forwardMapping = getMapping(minima1, minima2, seg2)
	backwardMapping = getMapping(minima2, minima1, seg1)
	strongForwardMappings, weakForwardMappings = getStrongAndWeakMappings(forwardMapping, backwardMapping)
	strongBackwardMappings, weakbackwardMappings = getStrongAndWeakMappings(backwardMapping, forwardMapping)
	writeMappings(strongForwardMappings, weakForwardMappings, weakbackwardMappings, outputFile, l1, loc1, l2, loc2)
