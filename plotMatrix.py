import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

fileDir = sys.argv[1]
outDir = os.path.join(fileDir, 'Output')
distDir = os.path.join(outDir, 'DistanceMatrices')


def getInputFiles(fileDir):
    files = os.listdir(fileDir)
    fileList = []
    for i in files:
        if 'Infinity' in i or 'PD' in i or "SF" in i or "Edit" in i or 'dist' in i:
            fileList.append(i)
    return fileList

matricesName = getInputFiles(distDir)
idx = fileDir.split('_')[-1].split('/')[0]
for mtx in matricesName:
    FileDir = os.path.join(distDir, mtx)
    M = np.loadtxt(FileDir)
    fig,ax = plt.subplots()
    vMin=0
    vMax=280
    treeType = ''
    distType = ''
    if 'st' in mtx:
        treeType = 'st'
    if 'mt' in mtx:
        treeType = 'mt'
    
    if "wD" in mtx:
        distType = 'wd'
        vMax=10
    if 'bD' in mtx:
        distType = 'bd'
        vMax=2.5
    if 'Infinity' in mtx:
        distType = 'IL'
        vMax=4.5
    
    #plt.imshow(M, cmap=cm.jet,alpha=0.75,vmin=0, vmax=vMax)
    print (M.shape)
    plt.imshow(M, cmap=plt.get_cmap('viridis'),alpha=0.75)
    #plt.imshow(M, cmap=cm.jet,alpha=0.75)
    plt.colorbar()
    ax.xaxis.tick_top()
    #plt.axis('off')
    #plt.title(mtx.split('.')[0])
    print (mtx.split('.')[0])
    pngName = str(idx)+treeType+'_'+distType + '.png'
    if 'SF' in mtx:
        pngName = str(idx) + 'SF.png'
        
    wingDir = "./wingMatrix/"
    pngDir = os.path.join(wingDir, pngName)
    plt.show()
    #fig.savefig(pngDir)
    
"""

for dataSet in dataSets:
    for scalarField in scalarFields:
            for IL in mappingStrategy:
                for extending_mode in extendingStrategy:
                    for PG in PGs:
                        FileDir = dataDir + 'Output_' + dataSet + '/' + scalarField + '/globalMapping/' + IL + '_' + extending_mode + '/distMatrixLInfinity.txt'
                        #FileDir = dataDir + 'Output_' + dataSet + '/' + scalarField + '/globalMapping/' + IL + '_' + extending_mode + '/distMatrixL2.txt'
                        M = np.loadtxt(FileDir)
                        plt.imshow(M, cmap=cm.jet,alpha=0.75)
                        plt.colorbar()
                        plt.show()
 """
