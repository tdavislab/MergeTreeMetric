from __future__ import division

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from numpy import linalg as LA
import numpy as np
import math
import os
import KernelPCA

n_components=2
eigen_solver='auto'
n_jobs=1

def precomputed_Isomap(D, n_neighbors):

    ALL_matrix = D

    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, metric='precomputed')
    
    print ("Generate distance matrix of NN graph...")
    dist_matrix_ = graph_shortest_path(kng,
                                       directed=False)
    
    G = dist_matrix_ ** 2
    G *= -0.5
    eigenxy, eigenval = KernelPCA.KernelPCA(n_components=2,
                                            kernel="precomputed",
                                            eigen_solver='auto',
                                            tol=0, max_iter=None,
                                            n_jobs=n_jobs).fit_transform(G)

    xy = eigenxy
    val = eigenval
  
    
    for i in range (0, 2):
        xy[:, i] = xy[:, i]*np.sqrt(val[i])

    return xy


def Tearing_Isomap(dist_matrix,ndims):

   
    G_ = dist_matrix 
    G = G_** 2
    G *= -0.5

    """
    xy1 = KernelPCA(n_components=ndims,
              kernel="precomputed",
              eigen_solver='auto',
              tol=0, max_iter=None,
              n_jobs=n_jobs).fit_transform(G)
    """
    eigenxy, eigenval = KernelPCA.KernelPCA(n_components=ndims,
                                            kernel="precomputed",
                                            eigen_solver='auto',
                                            tol=0, max_iter=None,
                                            n_jobs=n_jobs).fit_transform(G)
    
    xy = eigenxy
    val = eigenval
  
    
    for i in range (0, ndims):
        xy[:, i] = xy[:, i]*np.sqrt(val[i])

    return xy


