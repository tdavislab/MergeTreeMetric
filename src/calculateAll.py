# Author: Lin Yan <lynne.h.yan@gmail.com>

import sys
import os
import re
from copy import deepcopy

import numpy as np

import math
import copy
from scipy.optimize import linear_sum_assignment
import networkx as nx
from numpy import unravel_index

MAX_RADIUS = 99999999999999999999999999999999
FLUCT = 0.000001


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
      
def initialization(Trees, Tcnt):
    for i in range(0, Tcnt):
        Trees['tree-'+str(i)]={}
        Trees['tree-'+str(i)]['Nodes']=[]
        Trees['tree-'+str(i)]['Edges']=[]
        Trees['tree-'+str(i)]['IL-dist'] = []
        Trees['tree-'+str(i)]['local-dist'] = []
    return Trees

def load_data_2_trees(Trees, Tid, nodesFile, edgesFile):
    nodes = np.loadtxt(nodesFile)
    links = np.loadtxt(edgesFile).astype(int)
    # Rearrange nodes and links: moving leaves to the front and root to the last of the array.
    root = list(nodes[:, 3]).index(max(nodes[:,3]))
    if len(links.shape) == 1:
        links = np.array([links])
    
    leaves = find_leaves(links, nodes, root)
    lcnt = len(leaves)
    bvs = find_bvs(links, nodes)
    cps = list(np.concatenate((leaves, bvs)))
    cps.append(root)
    cps = [int(i) for i in cps]
    nodes, links = rearange_nodes_links_old(cps, nodes, links)

    Trees['tree-'+str(Tid)]['Nodes'] = nodes
    Trees['tree-'+str(Tid)]['Edges'] = links
    return Trees, lcnt


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def rearange_links(idx, links):
    for i in range(0, len(links)):
        links[i, 0] = idx[links[i, 0]]
        links[i, 1] = idx[links[i, 1]]
    for i in range (0, len(links)):
        links[i] = sorted(links[i])
    links = links[links[:,0].argsort()]   
    return links

def rearange_nodes_links_old(idx, nodes, links):
    nodes = nodes[idx,:]
    for i in range(0, len(links)):
        links[i, 0] = idx.index(links[i, 0])
        links[i, 1] = idx.index(links[i, 1])
    for i in range (0, len(links)):
        links[i] = sorted(links[i])
        
    links = links[links[:,0].argsort()]
    idx = update_idx_links(links[:,0], links[:,1])
    links = links[idx]
    return nodes, links

def update_idx_links(l1, l2):
    nidx = np.array(range(0, len(l1)))
    for i in set(list(l1)):
        if list(l1).count(i)>1:
            idx = [index for index, value in enumerate(l1) if value == i]
            new = np.array(l2[idx]).argsort()
            nidx[idx] = nidx[np.array(idx)[new]]
    return nidx

def convert_to_cmp(value, mn, mx):
    return int((value-mn)/mx*256)

def plot_2d_mt(nodes, links):
    mx = max(nodes[:,2])
    mn = min(nodes[:,2])
    colors = plt.cm.jet(np.linspace(0,1,256))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for i in range (0, len(links)):
        c_val = convert_to_cmp((nodes[links[i][0],2]+nodes[links[i][1],2])/2, mn, mx)
        ax.plot([nodes[links[i][0]][0], nodes[links[i][1]][0]], [nodes[links[i][0]][1],nodes[links[i][1]][1]], color = colors[c_val])
    ax.scatter(nodes[:,0],nodes[:, 1],s=40, c=nodes[:,2], cmap=plt.cm.get_cmap("jet"))
    plt.show()

def plot_2d_projection(Y, title=None):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cs = ax.scatter(Y[:,0], Y[:, 1],s=40, c=range(len(Y)), cmap=plt.cm.get_cmap("jet"))
    #ax.set_title(title)
    fig.colorbar(cs)
    plt.show()

def compute_sorted_index(leaves, ncnt):
    idx = []
    for i in range(0, len(leaves)):
        idx.append(int(leaves[i]))
    
    for i in range(0, ncnt):
        if i not in leaves:
            idx.append(int(i))
    return idx    
    
def mapping_leaves(nodes1, links1, nodes2, links2, lcnt, ED):
    #PD1 = calculatePersistenceDiagram(nodes1, links1)
    #PD2 = calculatePersistenceDiagram(nodes2, links2)
    dist = np.zeros((lcnt, lcnt))
    for i in range(0, lcnt):
        for j in range(0, lcnt):
            #idx1 = list(PD1[:,0]).index(nodes1[i][3])
            #idx2 = list(PD2[:,0]).index(nodes2[j][3])
            E_dist = np.linalg.norm(nodes1[i][0:3]-nodes2[j][0:3])
            #T_dist = np.linalg.norm(PD1[idx1]-PD2[idx2])
            #dist[i][j] = ED*E_dist+(1-ED)*T_dist
            dist[i][j] = E_dist
    row_ind, col_ind = linear_sum_assignment(dist)
    idx = np.zeros(lcnt).astype(int)-1
    for i in range(0, len(col_ind)):
        idx[col_ind[i]]=i
    leaves1 = np.array(idx)
    idx1 = compute_sorted_index(leaves1, len(nodes1))    
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
    return nodes1, links1

def add_leaves_links_new(nodes, links, idx):
    links = np.concatenate((links, [[idx, len(nodes)]]), axis=0)
    nodes = np.concatenate((nodes, [nodes[idx]]), axis=0)
    return nodes, links

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

def calculatePersistenceDiagram(nodes, links):
    root = list(nodes[:, 3]).index(max(nodes[:,3]))
    leaves = list(find_leaves(links, nodes, root))
    scalars = list(nodes[leaves, 3])
    leaves = [x for _,x in sorted(zip(scalars,leaves))]
    PD = find_minLeaf_each_branch(links, nodes, root, leaves)
    return PD

def extend_leaves(nodes1, links1, lcnt1, nodes2, links2, lcnt2, ED):
    #PD1 = calculatePersistenceDiagram(nodes1, links1)
    #PD2 = calculatePersistenceDiagram(nodes2, links2)
    dist = np.zeros((lcnt1, lcnt2))
    for i in range(0, lcnt1):
        for j in range(0, lcnt2):
            #idx1 = list(PD1[:,0]).index(nodes1[i][3])
            #idx2 = list(PD2[:,0]).index(nodes2[j][3])
            E_dist = np.linalg.norm(nodes1[i][0:3]-nodes2[j][0:3])
            #T_dist = np.linalg.norm(PD1[idx1]-PD2[idx2])
            #dist[i][j] = ED*E_dist+(1-ED)*T_dist
            dist[i][j] = E_dist
    row_ind, col_ind = linear_sum_assignment(dist)
    idx = np.zeros(lcnt2).astype(int)-1
    maxv = dist.max()
    UK1idx = []
    UK2idx = []
    nleaves1 = []
    nleaves2 = []
    tmpcnt = 0
    for i in range(0, len(col_ind)):
        if dist[i,  col_ind[i]]<MAX_RADIUS:
            idx[col_ind[i]]=i
            tmpcnt += 1
    UKidx=[]
    Kidx=[]
    for i in range(0, lcnt2):
        if idx[i] == -1:
            UKidx.append(i)
        else:
            UK2idx.append(idx[i])
            Kidx.append(i)
            
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes2)])
    for i in range(0, len(links2)):
        E_dist = np.linalg.norm(nodes2[links2[i][0]][3]-nodes2[links2[i][1]][3])
        G.add_edge(links2[i][0],links2[i][1],weight=E_dist)
    distV2 = np.zeros((len(UKidx), len(Kidx)))    
    for i in range(0, len(UKidx)):
        for j in range(0, len(Kidx)):
            distV2[i, j] = (1-ED)*nx.shortest_path_length(G, source=UKidx[i], target=Kidx[j], weight='weight')+ED*np.linalg.norm(nodes2[UKidx[i]][[0,1]]-nodes2[Kidx[j]][[0,1]])

    G=nx.Graph()
    G.add_nodes_from([0, len(nodes1)])
    for i in range(0, len(links1)):
        E_dist = np.linalg.norm(nodes1[links1[i][0]][3]-nodes1[links1[i][1]][3])
        G.add_edge(links1[i][0],links1[i][1],weight=E_dist)
    distV1 = np.zeros((lcnt1, lcnt1))

    for i in range(0, lcnt1):
        for j in range(0, lcnt1):
            distV1[i, j] = (1-ED)*nx.shortest_path_length(G, source=i, target=UK2idx[j], weight='weight')+ED*np.linalg.norm(nodes1[i][[0,1]]-nodes1[UK2idx[j]][[0,1]])
           
    idx = list(idx)
    for i in range(0, len(UKidx)):
        tmp = find_min_square_dist(distV2[i], distV1)
        #tmp = np.argmin(dist[:, UKidx[i]])
        idx[UKidx[i]]=len(nodes1)
        nodes1, links1 = add_leaves_links_new(nodes1, links1, tmp)

    leaves1 = np.array(idx)
    
    idx1 = compute_sorted_index(leaves1, len(nodes1))    
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
    return nodes1, links1


def get_tree_dist(lcnt, nodes, links):
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes)])
    for i in range(0, len(links)):
        E_dist = abs(nodes[links[i][0]][3]-nodes[links[i][1]][3])
        #E_dist = np.linalg.norm(nodes[links[i][0]][[0,2]]-nodes[links[i][1]][[0,2]])
        G.add_edge(links[i][0],links[i][1],weight=E_dist)
        #G.add_edge(links[i][0],links[i][1],weight=1)
    dist = np.zeros((lcnt, lcnt))
    for i in range(lcnt):
        for j in range (0, lcnt):
            dist[i,j] =nx.shortest_path_length(G, source=i, target=j, weight='weight')
    return dist

def get_tree_dist_StrongMapping(idx, nodes, links):
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes)])
    for i in range(0, len(links)):
        E_dist = abs(nodes[links[i][0]][3]-nodes[links[i][1]][3])
        #E_dist = np.linalg.norm(nodes[links[i][0]][[0,2]]-nodes[links[i][1]][[0,2]])
        G.add_edge(links[i][0],links[i][1],weight=E_dist)
        #G.add_edge(links[i][0],links[i][1],weight=1)
    dist = np.zeros((len(nodes), len(idx)))
    for i in range(len(nodes)):
        for j in range(len(idx)):
            dist[i,j] =nx.shortest_path_length(G, source=i, target=idx[j], weight='weight')
    return dist

def square_dist(a, b):
    tmp = 0
    for i in range(0, len(a)):
        tmp = (a[i]-b[i])**2+tmp
    return tmp

def addDummyUnderLeaf(nodes, links, idx):
    nnodes = deepcopy(nodes).tolist()
    nlinks = deepcopy(links).tolist()
    x = nodes[idx][0]
    y = nodes[idx][1]
    z = nodes[idx][2]
    SF = nodes[idx][3]
    nnodes.append([x, y, z, SF+FLUCT])
    nnodes.append([x, y, z, SF])
    linkId, leaf = [[i, link.index(idx)] for i, link in enumerate(nlinks) if idx in link][0]
    parent = nlinks[linkId][1-leaf]
    nlinks.remove([nlinks[linkId][0], nlinks[linkId][1]])
    nlinks.append([parent, len(nodes)])
    nlinks.append([idx, len(nodes)])
    nlinks.append([len(nodes), len(nodes)+1])
    return np.array(nnodes), np.array(nlinks), len(nodes)+1

def getDistVectorOfDummy(nodes, links, lcnt, SF, link):
    s1 = nodes[link[0]][3]
    s2 = nodes[link[1]][3]
    x1 = nodes[link[0]][0]
    x2 = nodes[link[1]][0]
    y1 = nodes[link[0]][1]
    y2 = nodes[link[1]][1]
    z1 = nodes[link[0]][2]
    z2 = nodes[link[1]][2]
    
    if s1 == s2:
        x = (x1+x2)/2
        y = (y1+y2)/2
        z = (z1+z2)/2
    else:
        x = (x1*(s2-SF)-x2*(s1-SF))/(s2-s1)
        y = (y1*(s2-SF)-y2*(s1-SF))/(s2-s1)
        z = (z1*(s2-SF)-z2*(s1-SF))/(s2-s1)
    nnodes = deepcopy(nodes).tolist()
    nlinks = deepcopy(links).tolist()
    nnodes.append([x, y, z, SF+FLUCT])
    nlinks.remove([link[0], link[1]])
    idx = len(nnodes)-1
    nlinks.append([link[0], idx])
    nlinks.append([link[1], idx])

    G=nx.Graph()
    G.add_nodes_from([0, len(nnodes)])
    for i in range(0, len(nlinks)):
        E_dist = np.linalg.norm(nnodes[nlinks[i][0]][3]-nnodes[nlinks[i][1]][3])
        G.add_edge(nlinks[i][0],nlinks[i][1],weight=E_dist)

    dist = []
    if type(lcnt) is list:
            for i in range(len(lcnt)):
                dist.append(nx.shortest_path_length(G, source=idx, target=lcnt[i], weight='weight'))
    else:
        for i in range(lcnt):    
            dist.append(nx.shortest_path_length(G, source=idx, target=i, weight='weight'))

    nnodes.append([x, y, z , SF])
    nlinks.append([idx, idx+1])
        
    return dist, nnodes, nlinks, idx+1

def addDummyUpRoot(nodes, links, lcnt, SF, root):
    x = nodes[root][0]
    y = nodes[root][1]
    z = nodes[root][2]
    nnodes = deepcopy(nodes).tolist()
    nlinks = deepcopy(links).tolist()
    nnodes.append([x, y, z, SF+FLUCT])
    nnodes.append([x, y, z,  SF])
    nlinks.append([root, len(nnodes)-2])
    nlinks.append([len(nnodes)-2, len(nnodes)-1])
    return nnodes, nlinks, len(nnodes)-1

def extend_leaves_dmyNodes(nodes1, links1, lcnt1, nodes2, links2, lcnt2, ED):
    #PD1 = calculatePersistenceDiagram(nodes1, links1)
    #PD2 = calculatePersistenceDiagram(nodes2, links2)
    dist = np.zeros((lcnt1, lcnt2))
    for i in range(0, lcnt1):
        for j in range(0, lcnt2):
            #idx1 = list(PD1[:,0]).index(nodes1[i][3])
            #idx2 = list(PD2[:,0]).index(nodes2[j][3])
            E_dist = np.linalg.norm(nodes1[i][0:3]-nodes2[j][0:3])
            #T_dist = np.linalg.norm(PD1[idx1]-PD2[idx2])
            #dist[i][j] = ED*E_dist+(1-ED)*T_dist
            dist[i][j] = E_dist
    row_ind, col_ind = linear_sum_assignment(dist)
    idx = np.zeros(lcnt2).astype(int)-1
    
    dist1 = get_tree_dist(lcnt1, nodes1, links1)
    dist2 = get_tree_dist(lcnt2, nodes2, links2)

      
    maxv = dist.max()
    UK1idx = []
    UK2idx = []
    nleaves1 = []
    nleaves2 = []
    tmpcnt = 0
    for i in range(0, len(col_ind)):
        if dist[i,  col_ind[i]]<MAX_RADIUS:
            idx[col_ind[i]]=i
            tmpcnt += 1

    UKidx=[]
    Kidx=[]

    uSF = []
    uDM = []
    maxSF = max(nodes1[:,3])
    minSF = min(nodes1[:,3])
    
    val = [x for i, x in enumerate(idx) if x > -1]
    index = np.array([i for i, x in enumerate(idx) if x > -1])
    index = list(index[sorted(range(len(val)), key=lambda k: val[k])])

    
    for i in range(0, lcnt2):
        if idx[i] == -1:
            UKidx.append(i)
            uSF.append(nodes2[i][3])
            uDM.append(dist2[i][index])
        else:
            UK2idx.append(idx[i])
            Kidx.append(i)
    for i in range(len(uSF)):
        minDist = MAX_RADIUS
        if uSF[i] >= maxSF:
            root = np.argmax(nodes1[:,3])
            nnodes, nlinks, nleave = addDummyUpRoot(nodes1, links1, lcnt1, uSF[i], root)
        elif uSF[i] <= minSF:
            tmpLeaf = -1
            for j in range(lcnt1):
                distDiff =  square_dist(dist1[j], uDM[i])
                if distDiff < minDist:
                    minDist = distDiff
                    tmpLeaf = j
            nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)
        else:
            tmpLeaf = -1
            for j in range(len(links1)):
                s1 = nodes1[links1[j][0]][3]
                s2 = nodes1[links1[j][1]][3]
                if (uSF[i]-s1) * (uSF[i]-s2) <= 0:
                    distTmp, nodesTmp, linksTmp, newLeave = getDistVectorOfDummy(nodes1, links1, lcnt1, uSF[i], links1[j])
                    distDiff =  square_dist(distTmp, uDM[i])
                    if distDiff < minDist:
                        minDist = distDiff
                        nnodes = nodesTmp
                        nlinks = linksTmp
                        nleave = newLeave
            for j in range(lcnt1):
                if nodes1[j][3] > uSF[i]:
                    distDiff =  square_dist(dist1[j], uDM[i])
                    if distDiff < minDist:
                        minDist = distDiff
                        tmpLeaf = j
            if tmpLeaf > -1:
                nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)
                        
        idx[UKidx[i]] = nleave
        nodes1 = np.array(nnodes)
        links1 = np.array(nlinks)

    leaves1 = np.array(idx)
    idx1 = compute_sorted_index(leaves1, len(nodes1))
  
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
    return nodes1, links1

def find_min_distance(l, M, lcnt):
    minD = 9999999999999;
    tmp = -1
    for i in range(lcnt):
        E_dist = np.linalg.norm(l-M[i])
        if  E_dist< minD:
            minD = E_dist
            tmp = i
    return tmp
    

def find_min_square_dist(l, M):
    minD = 9999999999999;
    tmp = -1
    for i in range(len(M)):
        E_dist = np.linalg.norm(l-M[i])
        if  E_dist< minD:
            minD = E_dist
            tmp = i
    return tmp
            
    

def calculate_ultra_M(lcnt, nodes, links):
    aj_matrix = np.zeros((lcnt, lcnt))
    idx_matrix = np.zeros((lcnt, lcnt))
    for i in range(lcnt):
        idx_matrix[i, i] = i
    
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes)])
    for i in range(0, len(links)):
        G.add_edge(links[i][0],links[i][1],weight=1)
    for i in range(0, lcnt):
        for j in range (i, lcnt):
            path = shortest_path(G, i, j)           
            aj_matrix[i,j] = nodes[path[np.argmax(nodes[path,3])],3]
            idx_matrix[i,j] = path[np.argmax(nodes[path,3])]
    return aj_matrix, idx_matrix

def shortest_path(G, source, target):
    path = nx.dijkstra_path(G,source, target)                
    return path

def infinity_norm(M1, M2):
    M1 = fill_sym_matrix(M1)
    M2 = fill_sym_matrix(M2)
   
    M = np.absolute(M1-M2)
    IL_dist = M.max()
    idx = (unravel_index(M.argmax(), M.shape))
    return IL_dist, idx

def L1_norm(M1, M2):
    M1 = fill_sym_matrix(M1)
    M2 = fill_sym_matrix(M2)
    M = M1-M2
    M = M.flatten()
    for i in range(len(M)):
        if M[i] < 0:
            M[i] = -M[i]
    return sum(M)

def L2_norm(M1, M2):
    M1 = fill_sym_matrix(M1)
    M2 = fill_sym_matrix(M2)
    M = M1-M2
    M = M.flatten()
    for i in range(len(M)):
        M[i] = M[i] ** 2
    return sum(M) ** 0.5


def fill_sym_matrix(M):
    for i in range(0, len(M)):
        for j in range(0, i):
            M[i][j] = M[j][i]
    return M

def calculate_tree_dist(T1, T2, l1, l2, ED, extending_mode):
    nodes1 = copy.deepcopy(T1['Nodes'])
    nodes2 = copy.deepcopy(T2['Nodes'])
    links1 = copy.deepcopy(T1['Edges'])
    links2 = copy.deepcopy(T2['Edges'])

    if l1 == l2:
       nodes2, links2 = mapping_leaves(nodes2, links2, nodes1, links1, l1, ED)
    else:
        if l1 < l2:
            if extending_mode == "dmyVert":
                nodes1, links1 = extend_leaves_dmyNodes(nodes1, links1, l1, nodes2, links2, l2, ED)
            else:
                nodes1, links1 = extend_leaves(nodes1, links1, l1, nodes2, links2, l2, ED)
            l1 = l2
        else:
            if extending_mode == "dmyVert":
                nodes2, links2 = extend_leaves_dmyNodes(nodes2, links2, l2, nodes1, links1, l1, ED)
            else:
                nodes2, links2 = extend_leaves(nodes2, links2, l2, nodes1, links1, l1, ED)
            l2 = l1
    M1, M1_idx = calculate_ultra_M(l1, nodes1, links1)
    M2, M2_idx = calculate_ultra_M(l2, nodes2, links2)

    IL_distance, idx =  infinity_norm(M1, M2)
    
    return IL_distance, L1_norm(M1,M2), L2_norm(M1, M2), idx

def calculate_tree_dist_from_Ms(M1, M2, M1_idx, M2_idx):
    IL_dist, idx = infinity_norm(M1, M2)
    idx_forward = int(M2_idx[idx[0], idx[1]])
    idx_backward = int(M1_idx[idx[0], idx[1]])
    return IL_dist, L1_norm(M1,M2), L2_norm(M1, M2), idx_forward, idx_backward


def calculate_tree_matrices(T1, T2, l1, l2, ED, extending_mode, idx, fileDir):
    nodes1 = copy.deepcopy(T1['Nodes'])
    nodes2 = copy.deepcopy(T2['Nodes'])
    links1 = copy.deepcopy(T1['Edges'])
    links2 = copy.deepcopy(T2['Edges'])
    if l1 == l2:
       nodes2, links2 = mapping_leaves(nodes2, links2, nodes1, links1, l1, ED)
       T2['Nodes'] = nodes2
       T2['Edges'] = links2

    else:
       if l1 < l2:
            if extending_mode == "dmyVert":
                nodes1, links1 = extend_leaves_dmyNodes(nodes1, links1, l1, nodes2, links2, l2, ED)
            else:
                nodes1, links1 = extend_leaves(nodes1, links1, l1, nodes2, links2, l2, ED)
            l1 = l2
            T1['Nodes'] = nodes1
            T1['Edges'] = links1
       else:
            if extending_mode == "dmyVert":
                nodes2, links2 = extend_leaves_dmyNodes(nodes2, links2, l2, nodes1, links1, l1, ED)
            else:
                nodes2, links2 = extend_leaves(nodes2, links2, l2, nodes1, links1, l1, ED)
            l2 = l1
            if 'Gaussian_rotated' in fileDir:
                if idx == 0 or idx == 1:
                    nodes2[1] = nodes2[2]
                    links2[0][1] = 3
                    links2[1][0] = 1
                    links2[1][1] = 2
            
            T2['Nodes'] = nodes2
            T2['Edges'] = links2

    M1, M1_idx = calculate_ultra_M(l1, nodes1, links1)
    M2, M2_idx = calculate_ultra_M(l2, nodes2, links2)
    return M1, M2, nodes1, nodes2, M1_idx, M2_idx

def addDummyForwardLeaves(nodes, links, idx, nidx1, nidx2):
    nodes1 = copy.deepcopy(nodes)
    links1 = copy.deepcopy(links)
    for i in range(len(nidx2)):
        nodes1, links1, leave = addDummyUnderLeaf(nodes1, links1, nidx2[i])
        idx[nidx1[i]] = leave
        nodes1 = np.array(nodes1)
        links1 = np.array(links1)
    return nodes1, links1, idx

def findDummyLeavesInPivotTree(idx,bidx1,bidx2, nodes):
    nnodes = deepcopy(nodes).tolist()
    for i in range(len(bidx1)):
        x = nodes[bidx1[i]][0]
        y = nodes[bidx1[i]][1]
        z = nodes[bidx1[i]][2]
        SF = nodes[bidx1[i]][3]
        indices = [m for m, X in enumerate(nnodes) if X[0]==x and X[1]==y and X[2]==z and X[3]==SF]
        idx[indices[1]] = bidx2[i]
        nnodes[indices[1]][3] += FLUCT
    return idx

def findDummyLeavesInPivotTree_dmyNode(idx,bidx1,bidx2, nodes,nodes2, lcnt, l1, flagDmyNode):
    nnodes = deepcopy(nodes).tolist()
    for i in range(len(bidx1)):
        SF = nodes2[bidx2[i]][3]
        indices = [m for m, X in enumerate(nnodes) if m>lcnt-1 and m<l1 and X[3]==SF]
        if len(indices) > 0:
            nnodes[indices[0]][3] += FLUCT
            flagDmyNode = indices[0]
        else:
            idxtmp = copy.deepcopy(idx).tolist()
            flagDmyNode = idxtmp.index(-1)
        idx[flagDmyNode] = bidx2[i]  
    return idx, flagDmyNode

def findIndexOfNodes(nodes, node):
    return [i for i, X in enumerate(nodes) if abs(X[0]-node[0])<0.02 and abs(X[1]-node[1])<0.02 and abs(X[2]-node[2])<0.02][0]


def addDummyNodeBaseOnBackwordMapping(T1, T2, l1, l2, labelFile, dummyLeaves, dummyCnts):
    nodes1 = copy.deepcopy(T1['Nodes'])
    nodes2 = copy.deepcopy(T2['Nodes'])
    links1 = copy.deepcopy(T1['Edges'])
    links2 = copy.deepcopy(T2['Edges'])

    n1 = nodes1[:, 0:3].tolist()
    n2 = nodes2[:, 0:3].tolist()

    idx1 = []
    idx2 = []
    dataStart = False
    with open(labelFile) as f:
        for line in f:
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    idx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    idx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))               
            if (line == "Weak backward mappings:\n"):
                dataStart = True
    
    idx = np.argsort(idx1)
    idx1 =  np.array(idx1)[idx]
    idx2 = np.array(idx2)[idx]
    
    tmp = []
    tmpCnt = []
    leaves = range(l1)
    if len(idx1) > 0:
        for i in range(len(idx1)):            
            if idx1[i] not in dummyLeaves:
                dummyLeaves.append(idx1[i])
                dummyCnts.append(1)
                tmp.append(idx1[i])
                tmpCnt.append(1)
                nodes1, links1, leave = addDummyUnderLeaf(nodes1, links1, idx1[i])
                leaves.append(leave)
                l1 += 1
            elif idx1[i] in tmp:
                tmpCnt[tmp.index(idx1[i])] += 1
                if tmpCnt[tmp.index(idx1[i])] > dummyCnts[dummyLeaves.index(idx1[i])]:
                    dummyCnts[dummyLeaves.index(idx1[i])] += 1
                    nodes1, links1, leave = addDummyUnderLeaf(nodes1, links1, idx1[i])
                    leaves.append(leave)
                    l1 += 1
            else:
                tmp.append(idx1[i])
                tmpCnt.append(1)
            
        leaves1 = np.array(leaves)
        idx1 = compute_sorted_index(leaves1, len(nodes1))
        nodes1 = np.array(nodes1)
        links1 = np.array(links1)
        nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
    return nodes1, links1, l1, dummyLeaves, dummyCnts

def nodeInList(nodes, node):
    for X in nodes:
        if abs(X[0]-node[0])<0.02 and abs(X[1]-node[1])<0.02 and abs(X[2]-node[2])<0.02:
            return True
    return False

def addDummyNodeBaseOnBackwordMapping_dmyNode(T1, T2, l1, l2, labelFile):
    nodes1 = copy.deepcopy(T1['Nodes'])
    nodes2 = copy.deepcopy(T2['Nodes'])
    links1 = copy.deepcopy(T1['Edges'])
    links2 = copy.deepcopy(T2['Edges'])

    n1 = nodes1[:, 0:3].tolist()
    n2 = nodes2[:, 0:3].tolist()
    
    maxSF = max(nodes1[:,3])
    minSF = min(nodes1[:,3])
    
    idx1 = []
    idx2 = []
    dataStart = False
    with open(labelFile) as f:
        for line in f:
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    idx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    idx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))      
            if (line == "Weak backward mappings:\n"):
                dataStart = True
    
    idx = np.argsort(idx1)
    idx1 =  np.array(idx1)[idx]
    idx2 = np.array(idx2)[idx]

    sidx1 = []
    sidx2 = []
    dataStart = False
    with open(labelFile) as f:
        for line in f:
            if (line == "Weak forward mappings:\n"):
                dataStart = False
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    sidx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    sidx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))      
            if (line == "Strong mappings:\n"):
                dataStart = True

    dist1 = get_tree_dist_StrongMapping(sidx1, nodes1, links1)
    dist2 = get_tree_dist_StrongMapping(sidx2, nodes2, links2)

    leaves = range(l1)
    tt = -1
    for i in range(len(idx2)):
        SF = nodes2[idx2[i],3]
        minDist = MAX_RADIUS
        nleave = -1
        if SF >= maxSF:
            root = np.argmax(nodes1[:,3])
            nnodes, nlinks, nleave = addDummyUpRoot(nodes1, links1, l1, SF, root)
            maxSF = SF
        elif SF <= minSF:
            tmpLeaf = -1
            for j in range(l1):
                distDiff =  square_dist(dist1[j], dist2[idx2[i]])
                if distDiff < minDist:
                    minDist = distDiff
                    tmpLeaf = j
            nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)  
        else:
            tmpLeaf = -1
            for j in range(len(links1)):
                s1 = nodes1[links1[j][0]][3]
                s2 = nodes1[links1[j][1]][3]
                if (SF-s1) * (SF-s2) <= 0:
                    distTmp, nodesTmp, linksTmp, newLeave = getDistVectorOfDummy(nodes1, links1, sidx1, SF, links1[j])
                    distDiff =  square_dist(distTmp, dist2[idx2[i]])
                    if distDiff < minDist:
                        minDist = distDiff
                        nnodes = nodesTmp
                        nlinks = linksTmp
                        nleave = newLeave
            for j in range(l1):
                if nodes1[j][3] > SF:
                    distDiff =  square_dist(dist1[j], dist2[idx2[i]])
                    if distDiff < minDist:
                        minDist = distDiff
                        tmpLeaf = j
                        
            if tmpLeaf > -1:
                nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)
        leaves.append(nleave)     
        l1 = l1+1
        nodes1 = np.array(nnodes)
        links1 = np.array(nlinks)
       
    leaves1 = np.array(leaves)
    idx1 = compute_sorted_index(leaves1, len(nodes1))
    nodes1 = np.array(nodes1)
    links1 = np.array(links1)
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
    return nodes1, links1, l1



def addDummyNodeToPivotTree(Trees, lcnts, pivot, inputFiles, labelFileDir, extending_mode):
    dummyLeaves = []
    dummyCnts = []
    for i in range(len(lcnts)):
        if i != pivot:
            labelFileName = 'label_' + str(int(re.search(r'\d+', inputFiles[i]).group()))  + '.txt'
            labelFile = os.path.join(labelFileDir, labelFileName)
            if extending_mode == "dmyLeaf":
                Trees['tree-'+str(pivot)]['Nodes'], Trees['tree-'+str(pivot)]['Edges'], lcnts[pivot], dummyLeaves, dummyCnts = addDummyNodeBaseOnBackwordMapping(Trees['tree-'+str(pivot)], Trees['tree-'+str(i)], lcnts[pivot], lcnts[i],  labelFile, dummyLeaves, dummyCnts)
            else:
                Trees['tree-'+str(pivot)]['Nodes'], Trees['tree-'+str(pivot)]['Edges'], lcnts[pivot] = addDummyNodeBaseOnBackwordMapping_dmyNode(Trees['tree-'+str(pivot)], Trees['tree-'+str(i)], lcnts[pivot], lcnts[i],  labelFile)

    return Trees['tree-'+str(pivot)], lcnts[pivot]


def extend_leaves_MP(nodes1, links1, lcnt1, nodes2, links2, lcnt2,  UKidx, Kidx, UK2idx, idx):
    ED = 0.5
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes2)])
    for i in range(0, len(links2)):
        E_dist = np.linalg.norm(nodes2[links2[i][0]][3]-nodes2[links2[i][1]][3])
        G.add_edge(links2[i][0],links2[i][1],weight=E_dist)
    distV2 = np.zeros((len(UKidx), len(Kidx)))

    for i in range(0, len(UKidx)):
        for j in range(0, len(Kidx)):
            distV2[i, j] = (1-ED)*nx.shortest_path_length(G, source=UKidx[i], target=Kidx[j], weight='weight')+ED*np.linalg.norm(nodes2[UKidx[i]][[0,1]]-nodes2[Kidx[j]][[0,1]])

    G=nx.Graph()
    G.add_nodes_from([0, len(nodes1)])
    for i in range(0, len(links1)):
        E_dist = np.linalg.norm(nodes1[links1[i][0]][3]-nodes1[links1[i][1]][3])
        G.add_edge(links1[i][0],links1[i][1],weight=E_dist)
    distV1 = np.zeros((lcnt1, len(UK2idx)))

    for i in range(0, lcnt1):
        for j in range(0, len(UK2idx)):
            distV1[i, j] = (1-ED)*nx.shortest_path_length(G, source=i, target=UK2idx[j], weight='weight')+ED*np.linalg.norm(nodes1[i][[0,1]]-nodes1[UK2idx[j]][[0,1]])

    idx = list(idx)

    for i in range(0, len(UKidx)):
        tmp = find_min_square_dist(distV2[i], distV1)
        #tmp = np.argmin(dist[:, UKidx[i]])
        idx[UKidx[i]]=len(nodes1)
        nodes1, links1 = add_leaves_links_new(nodes1, links1, tmp)

    leaves1 = np.array(idx)
    
    idx1 = compute_sorted_index(leaves1, len(nodes1))    
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
            
    return nodes1, links1



def extend_leaves_MP_dmyNode(nodes1, links1, lcnt1, nodes2, links2, lcnt2,  UKidx, Kidx, UK2idx, idx):
    ED = 0.5
    G=nx.Graph()
    G.add_nodes_from([0, len(nodes2)])
    for i in range(0, len(links2)):
        E_dist = np.linalg.norm(nodes2[links2[i][0]][3]-nodes2[links2[i][1]][3])
        G.add_edge(links2[i][0],links2[i][1],weight=E_dist)
    distV2 = np.zeros((len(UKidx), len(Kidx)))
    
    for i in range(0, len(UKidx)):
        for j in range(0, len(Kidx)):
            distV2[i, j] = (1-ED)*nx.shortest_path_length(G, source=UKidx[i], target=Kidx[j], weight='weight')+ED*np.linalg.norm(nodes2[UKidx[i]][[0,1]]-nodes2[Kidx[j]][[0,1]])

    G=nx.Graph()
    G.add_nodes_from([0, len(nodes1)])
    for i in range(0, len(links1)):
        E_dist = np.linalg.norm(nodes1[links1[i][0]][3]-nodes1[links1[i][1]][3])
        G.add_edge(links1[i][0],links1[i][1],weight=E_dist)
    distV1 = np.zeros((lcnt1, len(UK2idx)))

    for i in range(0, lcnt1):
        for j in range(0, len(UK2idx)):
            distV1[i, j] = (1-ED)*nx.shortest_path_length(G, source=i, target=UK2idx[j], weight='weight')+ED*np.linalg.norm(nodes1[i][[0,1]]-nodes1[UK2idx[j]][[0,1]])

    maxSF = max(nodes1[:,3])
    minSF = min(nodes1[:,3])
    
    idx = list(idx)
    for i in range(0, len(UKidx)):
        SF = nodes2[UKidx[i],3]
        minDist = MAX_RADIUS
        if SF >= maxSF:
            root = np.argmax(nodes1[:,3])
            nnodes, nlinks, nleave = addDummyUpRoot(nodes1, links1, lcnt1, SF, root)
            maxSF = SF
        elif SF <= minSF:
            tmpLeaf = -1
            for j in range(lcnt1):
                distDiff =  square_dist(distV1[j], distV2[i])
                if distDiff < minDist:
                    minDist = distDiff
                    tmpLeaf = j
            nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)  
        else:
            tmpLeaf = -1
            for j in range(len(links1)):
                s1 = nodes1[links1[j][0]][3]
                s2 = nodes1[links1[j][1]][3]
                if (SF-s1) * (SF-s2) <= 0:
                    distTmp, nodesTmp, linksTmp, newLeave = getDistVectorOfDummy(nodes1, links1, UK2idx, SF, links1[j])
                    distDiff =  square_dist(distTmp, distV2[i])
                    if distDiff < minDist:
                        minDist = distDiff
                        nnodes = nodesTmp
                        nlinks = linksTmp
                        nleave = newLeave
            for j in range(lcnt1):
                if nodes1[j][3] > SF:
                    distDiff =  square_dist(distV1[j],  distV2[i])
                    if distDiff < minDist:
                        minDist = distDiff
                        tmpLeaf = j
            if tmpLeaf > -1:
                nnodes, nlinks, nleave = addDummyUnderLeaf(nodes1, links1, tmpLeaf)

        idx[UKidx[i]] = nleave
        nodes1 = np.array(nnodes)
        links1 = np.array(nlinks)

    leaves1 = np.array(idx)
    
    idx1 = compute_sorted_index(leaves1, len(nodes1))    
    nodes1, links1 = rearange_nodes_links_old(idx1, nodes1, links1)
            
    return nodes1, links1


def calculate_tree_matrices_MP(T1, T2, l1, l2, labelFile, extending_mode, pivotlcnt, flagDmyNode):
    nodes1 = copy.deepcopy(T1['Nodes'])
    nodes2 = copy.deepcopy(T2['Nodes'])
    links1 = copy.deepcopy(T1['Edges'])
    links2 = copy.deepcopy(T2['Edges'])

    n1 = nodes1[:, 0:3].tolist()
    n2 = nodes2[:, 0:3].tolist()
    
    idx1 = []
    idx2 = []
    dataStart = False
    with open(labelFile) as f:
        for line in f:
            if (line == "Weak forward mappings:\n"):
                dataStart = False
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    idx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    idx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))      
            if (line == "Strong mappings:\n"):
                dataStart = True

    nidx1 = []
    nidx2 = []
    with open(labelFile) as f:
        for line in f:
            if (line == "Weak backward mappings:\n"):
                dataStart = False
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    nidx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    nidx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))      
            if (line == "Weak forward mappings:\n"):
                dataStart = True

    bidx1 = []
    bidx2 = []
    with open(labelFile) as f:
        for line in f:
            if(dataStart):
                idxs = np.array(line.split()).astype(float)
                if (nodeInList(n1, [idxs[0], idxs[1], idxs[2]])):
                    bidx1.append(findIndexOfNodes(n1, [idxs[0], idxs[1], idxs[2]]))
                if (nodeInList(n2, [idxs[3], idxs[4], idxs[5]])):
                    bidx2.append(findIndexOfNodes(n2, [idxs[3], idxs[4], idxs[5]]))      
            if (line == "Weak backward mappings:\n"):
                dataStart = True
    
    idx = np.argsort(idx1)
    idx1 =  np.array(idx1)[idx]
    idx2 = np.array(idx2)[idx]
    
    idx = np.zeros(l1).astype(int)-1
    
    for i in range(len(idx1)):
        idx[idx1[i]] = idx2[i]

    if len(nidx1)>0:
        if extending_mode=='dmyLeaf':
            nodes2, links2, idx = addDummyForwardLeaves(nodes2, links2, idx, nidx1, nidx2)
            l2 += len(nidx1)

    idxtmp = copy.deepcopy(idx).tolist()
    if flagDmyNode == -1:
        if -1 in idxtmp:
            flagDmyNode = idxtmp.index(-1)
    
    if extending_mode=='dmyLeaf':
        if len(bidx1)>0:
            idx = findDummyLeavesInPivotTree(idx,bidx1,bidx2, nodes1)
            l2 -= len(bidx1)
    else:
        if len(bidx1)>0:
            idx, flagDmyNode = findDummyLeavesInPivotTree_dmyNode(idx,bidx1,bidx2, nodes1, nodes2, pivotlcnt, l1, flagDmyNode)
            l2 -= len(bidx1)

        
    UKidx = []
    Kidx = []
    UK2idx = []

    for i in range(l1):
        if idx[i] == -1:
            UKidx.append(i)
        else:
            UK2idx.append(idx[i])
            Kidx.append(i)
    if extending_mode=='dmyLeaf':
        nodes2, links2 = extend_leaves_MP(nodes2, links2, l2, nodes1, links1, l1, UKidx, Kidx, UK2idx, idx)
    else:
        nodes2, links2 = extend_leaves_MP_dmyNode(nodes2, links2, l2, nodes1, links1, l1, UKidx, Kidx, UK2idx, idx)
        
    l2 = l1

    M1, M1_idx = calculate_ultra_M(l1, nodes1, links1)
    M2, M2_idx = calculate_ultra_M(l2, nodes2, links2)
            
    return M1, M2, nodes1, nodes2, flagDmyNode, M1_idx, M2_idx
            
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
                leaves.append(elem[i])
    return np.array(leaves)

def find_bvs(links, nodes):
    n,dim = links.shape
    elems = links.reshape((1,n*dim))
    elem, cnt = np.unique(elems, return_counts=True)
    bvs = []
    lnklist = links.tolist()
    for i in range(0,len(elem)):
        if cnt[i]>2:
            bvs.append(elem[i])
    return np.array(bvs)

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def removeTranparent(img):
    x,y = img.shape
    tmp = np.zeros(y).tolist()
    img = img.tolist()
    while tmp in img:
        img.remove(tmp)
    img = np.array(img).transpose()
    x,y = img.shape
    tmp = np.zeros(y).tolist()
    img = img.tolist()
    while tmp in img:
        img.remove(tmp)
    return np.array(img).transpose()
    

def getScalarFieldMatrix(scalarFile, scalarField):
    if len(np.loadtxt(scalarFile).shape) == 1:
        return [np.loadtxt(scalarFile)]
    if scalarField == 'anisotropy':
        scalar =  np.loadtxt(scalarFile)[0]
    else:
        scalar =  np.loadtxt(scalarFile)[1]
    return [scalar]
 

def getIndexOfNodes(samples, l1, l2):
    index = []
    for i in range(len(samples)):
        index.append(l2.index(l1[samples[i]]))
    return index

def plot_matrix(M1, M2, M3):
    plt.imshow(M1)
    plt.colorbar()
    plt.show()

    plt.imshow(M2)
    plt.colorbar()
    plt.show()

    plt.imshow(M3)
    plt.colorbar()
    plt.show()
    
def savePairwiseDistanceMatrix(Trees, Tcnt, lcnts, IL, extending_mode, outDir, fileInfo, ED):
    distDir = os.path.join(outDir, 'DistanceMatrices')
    if IL == 'TD':
        ED = 0
    if IL == 'ED':
        ED = 1

    dist_matrix = np.zeros((Tcnt, Tcnt))
    dist_matrix_L1 = np.zeros((Tcnt, Tcnt))
    dist_matrix_L2 = np.zeros((Tcnt, Tcnt))
    do_indx_matrix = np.zeros((Tcnt, Tcnt))
    for i in range(Tcnt):
        for j in range(i+1, Tcnt):
            dist_matrix[i, j], dist_matrix_L1[i, j], dist_matrix_L2[i, j], idx= calculate_tree_dist(Trees['tree-'+str(i)], Trees['tree-'+str(j)], lcnts[i], lcnts[j], ED, extending_mode)
            dist_matrix[j, i] = dist_matrix[i, j]
            dist_matrix_L1[j, i] = dist_matrix_L1[i, j]
            dist_matrix_L2[j, i] = dist_matrix_L2[i, j]
            do_indx_matrix[j, i] = do_indx_matrix[i, j]
    outFile1 = os.path.join(distDir, 'distMatrixLInfinity' + fileInfo + '.txt')
    outFile2 = os.path.join(distDir, 'distMatrixL1' + fileInfo + '.txt')
    outFile3 = os.path.join(distDir, 'distMatrixL2' + fileInfo + '.txt')
    
    #plot_matrix(dist_matrix, dist_matrix_L1, dist_matrix_L2)
    
    np.savetxt(outFile1, dist_matrix, fmt='%.4f')
    np.savetxt(outFile2, dist_matrix_L1, fmt='%.4f')
    np.savetxt(outFile3, dist_matrix_L2, fmt='%.4f')

def savellToTxt(outFile, ll):
    with open(outFile, 'w') as f:
        for l in ll:
            f.write(str(l) + '\n')
    
    
def saveGlobalMappingInfo(fileDir, scalarField, inputFiles, Trees, Tcnt, lcnts, IL, extending_mode, outDir, fileInfo, flagGM, ED):

    distDir = os.path.join(outDir, 'DistanceMatrices')
    diagDir = os.path.join(outDir, 'Diagnose')
    if IL == 'TD':
        ED = 0
    if IL == 'ED':
        ED = 1
    dist_matrix = np.zeros((Tcnt, Tcnt))
    do_idx_matrix = np.zeros((Tcnt, Tcnt)).astype(int)
    do_idx_matrix_back = np.zeros((Tcnt, Tcnt)).astype(int)
    dist_matrix_L1 = np.zeros((Tcnt, Tcnt))
    dist_matrix_L2 = np.zeros((Tcnt, Tcnt))
    pivotlcnt = max(lcnts)
    pivot = lcnts.index(pivotlcnt)

    Ms = []
    maxmv = 0;
    minmv = 9999999;
    nodes = []
    lls = []
    M_idxs = []
    nodesOrign = []
    for i in range(Tcnt):
        Ms.append([])
        nodes.append([])
        lls.append([])
        M_idxs.append([])
        nodesOrign.append([])
        
    ## Add dummy nodes to pivot tree based on backward mapping in Morse mapping 
    if IL == "MP":
        interDir = os.path.join(fileDir,'IntermediateFiles')
        mpDir = os.path.join(interDir, "MorseMappingLabels")
        labelFileDir = os.path.join(mpDir, scalarField)
        Trees['tree-'+str(pivot)],  lcnts[pivot] = addDummyNodeToPivotTree(Trees, lcnts, pivot, inputFiles, labelFileDir, extending_mode)

    flagDmyNode = -1
    for i in range(pivot+1, Tcnt):
        if i != pivot:
            if IL == "MP":
                labelFileName = 'label_' + str(int(re.search(r'\d+', inputFiles[i]).group())) + '.txt'
                labelFile = os.path.join(labelFileDir,labelFileName)
                Mp, Mi, nodesp, nodesi, flagDmyNode, Mp_idx, Mi_idx = calculate_tree_matrices_MP(Trees['tree-'+str(pivot)], Trees['tree-'+str(i)], lcnts[pivot], lcnts[i], labelFile, extending_mode, pivotlcnt, flagDmyNode)
            else:
                #print (i)
                Mp, Mi, nodesp, nodesi, Mp_idx, Mi_idx = calculate_tree_matrices(Trees['tree-'+str(i-1)], Trees['tree-'+str(i)], lcnts[pivot], lcnts[i], ED, extending_mode, i, fileDir)
            tmp =  sorted(list(set(list(Mi.flatten()))))
            tmp1 = min([n for n in tmp  if n>0])
            if minmv > tmp1:
                minmv = tmp1
            if maxmv < Mi.max():
                maxmv = Mi.max()
            if len(Ms[pivot])==0:
                tmp =  sorted(list(set(list(Mp.flatten()))))
                tmp1 = min([n for n in tmp  if n>0])
                Ms[pivot] = Mp
                M_idxs[pivot] = Mp_idx
                nodesOrign[pivot] = nodesp
                nodespNew, llp = rearangeNodes(nodesp, lcnts[pivot])
                nodes[pivot] = nodespNew
                lls[pivot] = llp
                if maxmv < Mp.max():
                    maxmv = Mp.max()
                if minmv > tmp1:
                    minmv = tmp1
            Ms[i] = Mi
            M_idxs[i] = Mi_idx
            nodesOrign[i] = nodesi
            nodesiNew, lli = rearangeNodes(nodesi, lcnts[pivot])
            nodes[i] = nodesiNew
            lls[i] = lli
            
    for i in range(pivot):
        if i != pivot:
            if IL == "MP":
                labelFileName = 'label_' + str(int(re.search(r'\d+', inputFiles[i]).group())) + '.txt'
                labelFile = os.path.join(labelFileDir, labelFileName)
                Mp, Mi, nodesp, nodesi, flagDmyNode, Mp_idx, Mi_idx = calculate_tree_matrices_MP(Trees['tree-'+str(pivot)], Trees['tree-'+str(i)], lcnts[pivot], lcnts[i], labelFile, extending_mode, pivotlcnt, flagDmyNode)
            else:
                #print (pivot-i)
                Mp, Mi, nodesp, nodesi, Mp_idx, Mi_idx = calculate_tree_matrices(Trees['tree-'+str(pivot-i)], Trees['tree-'+str(pivot-i-1)], lcnts[pivot], lcnts[pivot-i-1], ED, extending_mode, pivot-i-1, fileDir)
            tmp =  sorted(list(set(list(Mi.flatten()))))
            if minmv > tmp[1]:
                minmv = tmp[1]
            if maxmv < Mi.max():
                maxmv = Mi.max()
            if len(Ms[pivot])==0:
                tmp =  sorted(list(set(list(Mp.flatten()))))
                Ms[pivot] = Mp
                M_idxs[pivot] = Mp_idx
                nodesOrign[pivot] = nodesp
                nodespNew, llp = rearangeNodes(nodesp, lcnts[pivot])
                nodes[pivot] = nodespNew
                lls[pivot] = llp
                if maxmv < Mp.max():
                    maxmv = Mp.max()
                if minmv > tmp[1]:
                    minmv = tmp[1]
            Ms[pivot-i-1] = Mi
            M_idxs[pivot-i-1] = Mi_idx
            nodesOrign[pivot-i-1] = nodesi
            nodesiNew, lli = rearangeNodes(nodesi, lcnts[pivot])
            nodes[pivot-i-1] = nodesiNew
            lls[pivot-i-1] = lli
    pivotlcnt = lcnts[pivot]
    
    for i in range(Tcnt):
        for j in range(i+1, Tcnt):
            dist_matrix[i, j], dist_matrix_L1[i, j], dist_matrix_L2[i, j], do_idx_matrix[i, j], do_idx_matrix_back[i, j] = calculate_tree_dist_from_Ms(Ms[i], Ms[j],M_idxs[i], M_idxs[j])
            dist_matrix[j, i] = dist_matrix[i, j]
            dist_matrix_L1[j, i] = dist_matrix_L1[i, j]
            dist_matrix_L2[j, i] = dist_matrix_L2[i, j]

    outFile1 = os.path.join(distDir, 'distMatrixLInfinity' + fileInfo + '.txt')
    outFile2 = os.path.join(distDir, 'distMatrixL1' + fileInfo + '.txt')
    outFile3 = os.path.join(distDir, 'distMatrixL2' + fileInfo + '.txt')
    
    outFile4 = os.path.join(diagDir, 'transCritical' + fileInfo + '.txt')
    outFile5 = os.path.join(diagDir, 'transCriticalBack' + fileInfo + '.txt')

    idxNode = []
    idxNode_back = []
    for i in range(Tcnt-1):
        idxNode.append(do_idx_matrix[i, i+1])
        idxNode_back.append(do_idx_matrix_back[i, i+1])

    posNode = []
    posNode_back = []

    for i in range(Tcnt):
        if i == 0:
            posNode.append([-999, -999, -999, -999])
        else:
            posNode.append(nodesOrign[i][idxNode[i-1]])
    for i in range(Tcnt):      
        if i == Tcnt-1:
            posNode_back.append([-999, -999, -999, -999])
        else:
            posNode_back.append(nodesOrign[i][idxNode_back[i]])
            
    posNodes = np.array(posNode)
    posNodes_back = np.array(posNode_back)
    np.savetxt(outFile1, dist_matrix, fmt='%.4f')
    np.savetxt(outFile2, dist_matrix_L1, fmt='%.4f')
    np.savetxt(outFile3, dist_matrix_L2, fmt='%.4f')
    np.savetxt(outFile4, posNodes, fmt='%.6f')
    np.savetxt(outFile5, posNodes_back, fmt='%.6f')

    #plot_matrix(dist_matrix, dist_matrix_L1, dist_matrix_L2)
    
    if flagGM == 1:
        infoDir = os.path.join(outDir, 'LabelInfo')
        make_dir(infoDir)
        Info = np.array([pivotlcnt, minmv, maxmv])
        outFile4 = os.path.join(infoDir, 'info.txt')
        np.savetxt(outFile4, Info)

 
        for i in range(Tcnt):
            outFile5 = os.path.join(infoDir, 'M_'+str(i)+'.txt')
            np.savetxt(outFile5, Ms[i])
            outFile6 = os.path.join(infoDir, 'Nodes_'+str(i)+'.txt')
            np.savetxt(outFile6, nodes[i])
            outFile6 = os.path.join(infoDir, 'll_'+str(i)+'.txt')
            savellToTxt(outFile6, lls[i])


def PROJECTION(fileDir, scalarField, IL, extending_mode, PG, flagGM, ED):
    inputFiles = getInputFiles(fileDir)
    outDir =  os.path.join(fileDir, 'Output')
    
    Tcnt = len(inputFiles)
    # Initialize and load data to trees.
    Trees = {}
    Trees = initialization(Trees, Tcnt)
    tcnt = 0
    lcnts = []
    interDir = os.path.join(fileDir, 'IntermediateFiles')
    mtDir = os.path.join(interDir, 'MergeTrees')
    mtDir = os.path.join(mtDir, scalarField)
    txtDir = os.path.join(mtDir, 'TXTFormat')
    for inputFile in inputFiles:
        fileName =  "monoMesh_" + str(int(re.search(r'\d+', inputFile).group()))
        nodesFile =  os.path.join(txtDir, 'treeNodes_' + fileName + '.txt')
        edgesFile =  os.path.join(txtDir, 'treeEdges_' + fileName + '.txt')
        Trees, lcnt = load_data_2_trees(Trees, tcnt, nodesFile, edgesFile)
        lcnts.append(lcnt)
        tcnt += 1
        
    fileInfo = "_" +scalarField + "_" + PG + "_" + IL + "_" + extending_mode
    if PG == 'PM':
        savePairwiseDistanceMatrix(Trees, Tcnt, lcnts, IL, extending_mode, outDir, fileInfo, ED)
    if PG == "GM":
        saveGlobalMappingInfo(fileDir, scalarField, inputFiles, Trees, Tcnt, lcnts, IL, extending_mode, outDir, fileInfo, flagGM, ED)


def rearangeNodes(nodes, lcnt):
    nodes = nodes.tolist()
    newn = []
    ll = []
    types = []
    for i in range(len(nodes)):
        if i < lcnt:
            ty = 1
        else:
            ty = 0
        if nodes[i] not in newn:
            newn.append( nodes[i])
            ll.append([i])
            types.append(ty)
        else:
            idx = newn.index(nodes[i])
            if ty == types[idx]:
                ll[idx].append(i)
    return newn, ll
    

def loadProjectionPos(data, pos, posE, posI, posL1, posL2, fileNames):
    data['Proj']=[]
    data['ProjE']=[]
    data['ProjI']=[]
    data['ProjL1'] = []
    data['ProjL2'] = []
    for i in range(0, len(fileNames)):
        data['Proj'].append({"x": pos[i][0], "y":pos[i][1], "name": fileNames[i]})
        data['ProjE'].append({"x": posE[i][0], "y":posE[i][1], "name": fileNames[i]})
        data['ProjI'].append({"x": posI[i][0], "y":posI[i][1], "name": fileNames[i]})
        data['ProjL1'].append({"x": posL1[i][0], "y":posL1[i][1], "name": fileNames[i]})
        data['ProjL2'].append({"x": posL2[i][0], "y":posL2[i][1], "name": fileNames[i]})
    return data

def getMatrixFromJson(links, Tcnt):
    M = np.zeros((Tcnt, Tcnt))
    for i in range(len(links)):
        M[links[i]['source'], links[i]['target']] = links[i]['weight']
        M[links[i]['target'], links[i]['source']] = links[i]['weight']
    return M

def getNames(proj):
    names = []
    for i in range(len(proj)):
        names.append(proj[i]['name'])
    return names


def main():    
    fileDir = sys.argv[1]
    scalarField =  sys.argv[2]
    IL =  sys.argv[3]
    extending_mode = sys.argv[4]
    PG = sys.argv[5]
    flag = int(sys.argv[6])
    ED = float(sys.argv[7])
    PROJECTION(fileDir, scalarField, IL, extending_mode, PG, flag, ED)

if __name__ == '__main__':

    main()
  
  
   
