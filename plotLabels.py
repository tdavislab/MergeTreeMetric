import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re
from sklearn import manifold
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.patches as mpatches
from scipy import spatial
import sys
import copy
import json

fileDir = sys.argv[1]
outDir = os.path.join(fileDir, 'Output')
plotDir = os.path.join(outDir, 'LabelInfo')

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def getInputFiles(fileDir):
    files = os.listdir(fileDir)
    fileList = []
    for i in files:
        if "Nodes_" in i:
            fileList.append(i)
    order = []
    for i in range(len(fileList)):
        order.append(int(re.search(r'\d+', fileList[i]).group()))
    fileList = [x for _,x in sorted(zip(order,fileList))]
    return fileList

def plotMatrix(M):
    plt.imshow(M, cmap=cm.jet,alpha=0.75)
    plt.colorbar()
    plt.show()

def findXYTEXT(x,y, d, Y):
    tmp = -999999
    YY = np.array(copy.deepcopy(Y))
    idx = Y.index([x,y])
    YY = np.delete(YY, idx, axis=0)

    pos = np.array([[x+1.4*d, y], [x-1.4*d, y],[x, y+1.4*d], [x, y-1.4*d], [x+d, y-d], [x-d, y+d], [x+d, y+d], [x-d, y-d]])

    for i in range(len(pos)):
        dist, index = spatial.KDTree(YY).query(pos[i])
        if dist >tmp:
            tmp = dist
            xx = pos[i][0]
            yy = pos[i][1]

    Y.append([xx, yy])
    return xx, yy, Y
        
def plotLabels(Y, labels, figFile):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(Y[:,0], Y[:, 1], s=200, alpha=0.7, zorder=10,edgecolors='black')

    for j in range(len(labels)):
        if len(labels[j])==1:
            label = labels[j][0]
            if label < 3:
                ax.annotate(label, xy=(Y[j, 0], Y[j, 1]+2.5),zorder=20, ha="center",fontsize=15)
        else:
            label = str(labels[j][0])
            for k in range(1, len(labels[j])):
                label = label + ', ' + str(labels[j][k])
            ax.annotate(label, xy=(Y[j, 0], Y[j, 1]+2.5),zorder=20, ha="center", fontsize=15)
    plt.title(figFile.split("/")[-1].split(".")[0])            
    plt.xlim(60, 130)
    plt.ylim(60, 130)
    #plt.xlim(0, 480)
    #plt.ylim(0, 60)
    plt.axis('off')         
    plt.savefig(figFile)

if __name__ == '__main__':
    nodesFiles = getInputFiles(plotDir)
    mapping = "LabelInfo"
    figDir = os.path.join(outDir, "Figures")
    figDir = os.path.join(figDir,mapping)
    make_dir(figDir)
    for nodesFile in nodesFiles:
        nodes = np.loadtxt(os.path.join(plotDir, nodesFile))
        labelFile = os.path.join(plotDir, 'll_' + nodesFile.split('.')[0].split('_')[1] + '.txt')
        figFile = os.path.join(figDir, mapping + '_' + nodesFile.split('.')[0].split('_')[1] + '.png')
        with open(labelFile) as f:
            content = f.readlines()
        labels = [json.loads(x.strip()) for x in content]
        plotLabels(nodes, labels,figFile)

