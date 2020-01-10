#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:30:54 2019

@author: bolger
"""

from tslearn.utils import load_timeseries_txt
from tslearn.utils import to_time_series_dataset
from tslearn import metrics
import numpy as np
import networkx as nx

verb_oi = 'soulever'

filepath_full = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/velocitydata-"+verb_oi+".txt"
filepath_time = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/veltime-"+verb_oi+".txt"
filepath_samp = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/sampent-"+verb_oi+".txt"

velIn = load_timeseries_txt(filepath_full)
timeIn = load_timeseries_txt(filepath_time)
sampentIn = load_timeseries_txt(filepath_samp)

data_anal = sampentIn
data_anal = np.squeeze(data_anal)

dataset1 = to_time_series_dataset([data_anal])
dataset1b = np.squeeze(dataset1)

diffvel = metrics.cdist_dtw(dataset1b)   #calculate the cross-similarity matrix using DTW

G = nx.from_numpy_matrix(diffvel)
nx.convert_node_labels_to_integers(G,first_label=1)
v = nx.nodes(G)
nx.draw(G, nodecolor='r',edge_color='r',node_size=220, with_labels=True)

from matplotlib import pyplot as plt
from matplotlib import cm as cm


fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
labels = list(range(1,np.size(velIn,axis=0)+1))
cax = ax1.imshow(diffvel, interpolation='hanning', cmap=cmap, vmin=0, vmax=6)
ax1.set_xticks(np.arange(0, np.size(velIn,axis=0), 1));
ax1.set_yticks(np.arange(0, np.size(velIn,axis=0), 1));
ax1.set_xticklabels(labels,fontsize=8)
ax1.set_yticklabels(labels,fontsize=8)
ax1.set_xlabel("Trial Number")
ax1.set_ylabel("Trial Number")

ax1.grid(False)
plt.title('Cross-Similarity Matrix: '+verb_oi)
fig.colorbar(cax, ticks=np.arange(0, 6, 0.5))
plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage

fig = plt.figure()
linkage_matrix = linkage(diffvel, "ward",optimal_ordering=True)
dendrogram(linkage_matrix, labels=sujtitre , distance_sort = "descending")     #"1", "2","3", "4", "5","6", "7", "8","9", "10","11","12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
plt.title("Linkage- "+verb_oi)
plt.xlabel("Trial Nnumber")
plt.ylabel("Distance (Euclidean)")
plt.show()

#, "13", "14","15", "16", "17","18", "19"
###---Kmeans clustering of similarity matrix----
# 1. Center the similarity matrix
# 2. Take the eigen values of the matrix
# 3. Multiply the first two set of eigenvectors to the square root of diagonals of the eigenvalues to get the vectors.
# 4. Compute the kmeans 

from sklearn.cluster import KMeans

def center_simmatrix(H):
    nrows = np.size(H,axis=0)
    P = np.diag(H,k=0) - (1/nrows)
    x = -0.5 * P 
    x2 = np.multiply(x,H)
    Hcentd = np.multiply(x2,P)
    return Hcentd

matcntd = center_simmatrix(diffvel)
mateig, mateigv = np.linalg.eig(matcntd)
eigoi = mateig[0:2,]       # the eigen values
eigoi_z = eigoi.clip(0)   # every below 0 is set to 0
i = np.diag(np.sqrt(mateigv[0:2,]))
i[np.isnan(i)] = 0
X = np.multiply(eigoi_z,i)   # Multiply the two set of eigenvectors to the square root of the diagonals of the eigenvalues.
cinit = KMeans(n_clusters=3, random_state=0).fit_predict(X)



        

mds.tau <- function(H)
{
  n <- nrow(H)
   P <- diag(n) - 1/n
   return(-0.5 * P %*% H %*% P)
  }
  B<-mds.tau(fpdist)
  eig <- eigen(B, symmetric = TRUE)
  v <- eig$values[1:2]
  #convert negative values to 0.
 v[v < 0] <- 0
X <- eig$vectors[, 1:2] %*% diag(sqrt(v))
library(vegan)
km <- kmeans(X,centers= 5, iter.max=1000, nstart=10000) .
#embedding using MDS
cmd<-cmdscale(fpdist)