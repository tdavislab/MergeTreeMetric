import os
import re

import numpy as np
import sys
import networkx as nx
import subprocess
import csv
import matplotlib.pyplot as plt

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

def find_leaves(links, nodes, root):
    n,dim = links.shape
    elems = links.reshape((1,n*dim))
    elem, cnt = np.unique(elems, return_counts=True)
    leaves = []
    lnklist = links.tolist()
    for i in range(0,len(elem)):
        if cnt[i]==1:
            x = [x for x in lnklist if elem[i] in x]
            end = [y for y in x[0] if y != elem[i]][0]
            if elem[i] != root:
                leaves.append(elem[i].astype(int))
    return np.array(leaves)

def find_minLeaf_each_branch(links, nodes, root, leaves):
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes)])
    for i in range(0, len(links)):
        G.add_edge(links[i][0],links[i][1],weight=1)

    travelled = []
    PD = []
    for i in range(len(leaves)):
        path = nx.shortest_path(G, source=leaves[i], target=root)
        for j in range(len(path)):
            if path[j] not in travelled:
                travelled.append(path[j])
            else:
                PD.append([nodes[leaves[i], 3], nodes[int(path[j]), 3]])
                break
            if j == len(path) - 1:
                PD.append([nodes[leaves[i], 3], nodes[root, 3]])
    return np.array(PD)


def calculatePersistenceDiagram(nodes, links, outDir, outFile):
    outFileName = os.path.join(outDir, outFile)
    root = list(nodes[:, 3]).index(max(nodes[:,3]))
    leaves = list(find_leaves(links, nodes, root))
    scalars = list(nodes[leaves, 3])
    leaves = [x for _,x in sorted(zip(scalars,leaves))]
    PD = find_minLeaf_each_branch(links, nodes, root, leaves)
    np.savetxt(outFileName, PD, fmt='%.6f')

def calculatePDdist(mt_dir, inputFiles, outDir, treeType):
    lcnt = len(inputFiles)
    PD_dist = np.zeros((lcnt, lcnt))
    PD_dist_b = np.zeros((lcnt, lcnt))
    for i in range(lcnt):
        for j in range(i+1, lcnt):
            fileNamei = "monoMesh_" + str(int(re.search(r'\d+', inputFiles[i]).group()))
            fileNamej = "monoMesh_" + str(int(re.search(r'\d+', inputFiles[j]).group()))

            PDi = 'PD_' + fileNamei + '.txt'
            PDj = 'PD_' + fileNamej + '.txt'

            PDi = os.path.join(mt_dir, PDi)
            PDj = os.path.join(mt_dir, PDj)


            cmd = ['wasserstein_dist', PDi, PDj, '2']
            #wdist = subprocess.check_output('wasserstein_dist ' + PDi + ' ' + PDj + ' 2', shell=True).replace('\n', '')
            wdist = float(subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0])
            PD_dist[i, j] = wdist
            PD_dist[j, i] = wdist
        
            cmd = ['bottleneck_dist', PDi, PDj]
            #wdist = subprocess.check_output('wasserstein_dist ' + PDi + ' ' + PDj + ' 2', shell=True).replace('\n', '')
            bdist = float(subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0])
            PD_dist_b[i, j] = bdist
            PD_dist_b[j, i] = bdist


    fileName = 'wDistPD_' + treeType + '.txt'
    outFile = os.path.join(outDir, fileName)
    np.savetxt(outFile, PD_dist, fmt='%.6f')
 
    
    fileName = 'bDistPD_' + treeType + '.txt'
    outFile = os.path.join(outDir, fileName)
    np.savetxt(outFile, PD_dist_b, fmt='%.6f')
    

    
def calculateSFdist(fileDir, inputFiles, outDir):
    lcnt = len(inputFiles)
    sf_dist = np.zeros((lcnt, lcnt))
    scalars = []
    for i in range(lcnt):
        fileName = "scalar_" + str(int(re.search(r'\d+', inputFiles[i]).group())) + '.txt'
        scalars.append(np.loadtxt(os.path.join(fileDir, fileName)))
    for i in range(lcnt):
        for j in range(i+1, lcnt):
            sfdist = np.linalg.norm(scalars[i]-scalars[j])
            sf_dist[i, j] = sfdist
            sf_dist[j, i] = sfdist
    fileName = 'DistSF.txt'
    outFile = os.path.join(outDir, fileName)
    np.savetxt(outFile, sf_dist, fmt='%.2f')

def plotAllPC(mt_dir, inputFles, outFile, treeType, thr):
    mx = 0
    for j in range(len(inputFiles)):
        nbrs = []
        prts = []
        csvFile = mt_dir + "/PC_" + str(int(re.search(r'\d+', inputFiles[j]).group()))  + '.csv'
        with open(csvFile, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                prts.append(float(row[0]))
                nbrs.append(int(row[1]))
        prts.append(prts[-1]*2)
        if max(prts) > mx:
            mx = max(prts)
        nbrs.append(1)
        plt.plot(prts, nbrs, linewidth=1)
        
    plt.xlabel('Persistence')
    if treeType == 'st':
        plt.ylabel('Maxima Count')
    else:
        plt.ylabel('Minima Count')
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    if thr > 0:
        xtck = list(plt.xticks()[0]) + [thr]
        if thr < mx/20:
            xtck.remove(0)
        plt.xticks(xtck)
        plt.axvline(x=thr, color='k', linestyle='--', linewidth=0.8)
        plt.xlim(0-mx*0.05, mx*1.05)
        
    plt.savefig(outFile)
    plt.title("Persistence Curves")
    if thr == 0:
        plt.show()

def plot4PC(mt_dir, inputFiles, outFile, treeType, thr):
    mx = 0
    for j in range(len(inputFiles)):
        nbrs = []
        prts = []
        csvFile = mt_dir + "/PC_" + str(int(re.search(r'\d+', inputFiles[j]).group()))  + '.csv'
        with open(csvFile, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                prts.append(float(row[0]))
                nbrs.append(int(row[1]))
        prts.append(0.95668)
        nbrs.append(1)
        if max(prts) > mx:
            mx = max(prts)
        plt.plot(prts, nbrs, linewidth=1)
        
    plt.xlabel('Persistence')
    if treeType == 'st':
        plt.ylabel('Maxima Count')
    else:
        plt.ylabel('Minima Count')
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    xtck = list(plt.xticks()[0]) + [0.02, 0.04, 0.07, 0.1]
    if thr < mx/20:
        xtck.remove(0)
    plt.xticks(xtck, rotation='vertical')
    plt.axvline(x=0.02, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(x=0.04, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(x=0.07, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(x=0.1, color='k', linestyle='--', linewidth=0.8)
    plt.xlim(0-mx*0.05, mx*1.05)
        
    plt.title("Persistence Curves")
    plt.show()
    
    
if __name__ == '__main__':
    if len(sys.argv)<9 or len(sys.argv)>10:
        print("python script.py [Path to files] [Name of scalar field] [Mapping Strategy: TD/ED/ET/MP] [Extending Strategy: dmyLeaf/dmyVert] [Tree Type: jt/st] [Glabal or Pairwise Mapping: GM/PM] [Skip merge tree and morse smale calculation] [Output labelling result for global mapping] [threshold for simplification (optional)]")
        
    else:
        fileDir = sys.argv[1]
        scalarField = sys.argv[2]
        mappingStrategy = sys.argv[3]
        extendingStrategy = sys.argv[4]
        treeType = sys.argv[5]
        PG = sys.argv[6]
        flag = int(sys.argv[7])
        flag2 = sys.argv[8]
        if len(sys.argv)==10:
            threshold = float(sys.argv[9])

        ETparams = 0.5
        if mappingStrategy == "ET":
            ETparams = float(input("\n\nPlease enter lambda for hybrid mapping([0,1]): "))
        outDir = os.path.join(fileDir, 'Output')
        make_dir(outDir)

        interDir = os.path.join(fileDir, 'IntermediateFiles')
        make_dir(interDir)

        mtDir = os.path.join(interDir, 'MergeTrees')
        make_dir(mtDir)

        sfDir = os.path.join(interDir, 'ScalarFields')
        make_dir(sfDir)

        pcDir = os.path.join(interDir, 'PersistenceCurves')
        make_dir(pcDir)
        pc_dir = os.path.join(pcDir, treeType)      
        make_dir(pc_dir)

        pdDir = os.path.join(interDir, 'PersistenceDiagrams')
        make_dir(pdDir)
        pd_dir = os.path.join(pdDir, treeType)      
        make_dir(pd_dir)

        distDir = os.path.join(outDir, 'DistanceMatrices')
        make_dir(distDir)

        diagDir = os.path.join(outDir, 'Diagnose')
        make_dir(diagDir)

        plotDir = os.path.join(outDir, 'Figures')
        make_dir(plotDir)
        
        mt_dir = os.path.join(mtDir, treeType)      
        make_dir(mt_dir)
        inputFiles = getInputFiles(fileDir)

        if PG == 'PM' and mappingStrategy == "MP":
            print ("\n\nERROR: We cannot use pairwise mapping for Morse mapping!")
            sys.exit(-1)
        if PG == 'PM' and flag2 == 1:
            print ("\n\nERROR: We cannot provide labelling result for pairwise mapping!")
            sys.exit(-1)
     
  
        if flag == 0:
            if len(sys.argv)==9:
                for j in range(len(inputFiles)):
                    submitCommand = "pvpython ./src/pvComputePC.py " + fileDir+ inputFiles[j] + " PC_" + str(int(re.search(r'\d+', inputFiles[j]).group()))  + '.csv ' + pc_dir + '/ ' + scalarField + ' ' + treeType
                    os.system(submitCommand)

                plotAllPC(pc_dir, inputFiles, os.path.join(plotDir, 'PersistenceCurves_' + treeType + '.png'), treeType, 0)
                threshold = float(input("\n\nPlease enter persistence threshold: "))
                plotAllPC(pc_dir, inputFiles, os.path.join(plotDir, 'PersistenceCurves_' + treeType + '.png'), treeType, threshold)
           
            
         
            for j in range(len(inputFiles)):
                submitCommand = "pvpython ./src/pvComputeMT.py " + fileDir+ inputFiles[j] + " monoMesh_" + str(int(re.search(r'\d+', inputFiles[j]).group()))  + ' ' + mt_dir + '/ ' + str(threshold) + ' ' + scalarField + ' ' + treeType
                os.system(submitCommand)
   
            if mappingStrategy == "MP":
                mpDir = os.path.join(interDir, 'MorseMappingLabels')
                make_dir(mpDir)
                mpDir = os.path.join(mpDir, treeType)
                make_dir(mpDir)
                submitCommand = "vtkpython ./src/SaveMorseMapping.py " + fileDir + ' ' + str(threshold) + ' ' + treeType + ' ' + scalarField
                os.system(submitCommand)
            
            submitCommand = "python ./src/vtkToTXTFiles.py " + fileDir  + ' ' + treeType + ' ' + scalarField
            os.system(submitCommand)


            for j in range(len(inputFiles)):
            
                treeFileName = "monoMesh_" + str(int(re.search(r'\d+', inputFiles[j]).group()))
                edgeFile = 'treeEdges_' + treeFileName + '.txt'
                nodeFile = 'treeNodes_' + treeFileName + '.txt'

                txtDir = os.path.join(mt_dir, 'TXTFormat')
                nodes = np.loadtxt(os.path.join(txtDir, nodeFile))
                links = np.loadtxt(os.path.join(txtDir, edgeFile))
                if len(links.shape) == 1:
                    links = np.array([links])
                calculatePersistenceDiagram(nodes, links, pd_dir, 'PD_' + treeFileName + '.txt')

       
        if flag == 0:
            calculatePDdist(pd_dir, inputFiles, distDir, treeType)
            calculateSFdist(sfDir, inputFiles, distDir)

        submitCommand = "python ./src/calculateAll.py " + fileDir + ' ' +  treeType + ' ' + mappingStrategy + ' ' + extendingStrategy + ' ' + PG + ' ' + flag2 + ' ' + str(ETparams)
        os.system(submitCommand)

        if PG == "GM":
            submitCommand = "python ./src/visualizaTrans.py " + fileDir + ' ' +  treeType + ' ' + mappingStrategy + ' ' + extendingStrategy + ' ' + PG + ' ' + flag2
            os.system(submitCommand)
     
