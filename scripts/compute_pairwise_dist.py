import networkx as nx
import numpy as np
import pandas as pd 

from ete3 import Tree
import utilities.validate_trees as tree_val

import sys
import os

from tqdm import tqdm_notebook

import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from cassiopeia.TreeSolver.Node import Node

from tqdm import tqdm

import scanpy as sc

import seaborn as sns
sns.set_context("talk")
sns.set_style("whitegrid")

from skbio.tree import TreeNode
from scipy.spatial.distance import pdist, squareform

tree = TreeNode.read('/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/trees/lg3/lg3_tree_hybrid_priors.alleleThresh.processed.txt', 
                     'newick')
cm = pd.read_csv("/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/trees/lg3/lg3_character_matrix.alleleThresh.txt", sep='\t', index_col = 0)

# set the length of each branch to be 1
for n in tree.traverse():
    n.length = 1.0

print('computing phylo dists')
mat = tree.tip_tip_distances()
dmat = pd.DataFrame(mat.data, index = mat.ids, columns=mat.ids)

diam = np.max(dmat.values.flatten())
dmat_norm = dmat / diam

print('computing edit dists')
edit_dists = pdist(cm.loc[dmat.index], 
                        lambda u, v: Node('state-node', character_vec = u).get_modified_hamming_dist(Node('state-node', character_vec = v)))

edit_dists = pd.DataFrame(squareform(edit_dists), index = dmat.index, columns = dmat.index)

dmat.to_csv('lg3_phylodists.txt', sep='\t')
edit_dists.to_csv('lg3_editdists.txt', sep='\t')

# iu1 = np.triu_indices(edit_dists.shape[0])

# evec, tvec = edit_dists.values[iu1].flatten() / cm.shape[1], dmat_norm.values[iu1].flatten()

# #tree_val.dist_plotter(tvec, evec, '2D-Hist', diam=int(diam), n_targets = cm.shape[1], out_fp = 'test.pdf')
# hist = plt.hist2d(tvec, evec, bins=[int(diam) - 1, 20], cmap=cc.cm.CET_L19)
# plt.show()

# scat = plt.scatter(evec,tvec, c='r', marker='.', alpha=0.01)
# plt.show()