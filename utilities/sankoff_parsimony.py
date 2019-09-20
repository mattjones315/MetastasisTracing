import networkx as nx 
import pandas as pd 
import numpy as np
from functools import reduce
from tqdm import tqdm

from collections import OrderedDict, defaultdict

import scipy.stats as scs
import cassiopeia.TreeSolver.compute_meta_purity as cmp

def SANKOFF_SIGMA(s, sp):
    
    return 1 if s != sp else 0

def sankoff_fill_C(tree, root, C, node_to_i, label_to_j):

    md = cmp.get_max_depth(tree, root)

    # bottom up approach
    d = md - 1
    while d >= 0:
        
        internal_nodes = cmp.cut_tree(tree, d)
        for int_node in internal_nodes:
            children = list(tree.successors(int_node))
            
            # compute min cost for each child
            for s in range(C.shape[1]):
                opt_cost = 0
                for child in children:
                    i = node_to_i[child]
                    min_cost = np.min([SANKOFF_SIGMA(s, sp) + C[i, sp] for sp in range(C.shape[1])])
                    opt_cost += min_cost
                    
                C[node_to_i[int_node], s] = opt_cost
            
        d -= 1 
        
    return C

def sankoff(tree, possible_labels = ["LL", "RE", "RW", "M1", "M2", "Liv"]):
    
    root = [n for n in tree if tree.in_degree(n) == 0][0]
        
    tree = cmp.set_depth(tree, root)
    max_depth = cmp.get_max_depth(tree, root)
    tree = cmp.extend_dummy_branches(tree, max_depth)
    
    C = np.full((len(tree.nodes), len(possible_labels)), np.inf)
    
    # create maps betwene names and row/col of dynamic programming array, C
    bfs_postorder = [root]
    for e0, e1 in nx.bfs_edges(tree, root):
        bfs_postorder.append(e1)
    
    node_to_i = dict(zip(bfs_postorder, range(len(tree.nodes))))
    label_to_j = dict(zip(possible_labels, range(len(possible_labels))))

    # instantiate the dynamic programming matrix at leaves
    _leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
    for l in _leaves:
        label = tree.nodes[l]["label"]
        
        i, j = node_to_i[l], label_to_j[label]
        C[i, j] = 0
        
            
    C = sankoff_fill_C(tree, root, C, node_to_i, label_to_j)
            
    return C

def compute_q_sankoff(tree, factor, C, S = []):
    
    # index matrix C
    master_root = [n for n in tree if tree.in_degree(n) == 0][0]
    
    bfs_postorder = [master_root]
    for e0, e1 in nx.bfs_edges(tree, master_root):
        bfs_postorder.append(e1)
    
    node_to_i = dict(zip(bfs_postorder, range(len(tree.nodes))))
    label_to_j = dict(zip(S, range(len(S))))

    root = factor[0]
    children = factor[1]

    _t = np.zeros((len(S), len(S)))

    pars = np.amin(C[node_to_i[root],:])
    possible_labels = np.where(C[node_to_i[root], :] == pars)[0]

    for j in possible_labels:
        
        for c in children:

            c_i = node_to_i[c]
            value_arr = [SANKOFF_SIGMA(j, sp) + C[c_i, sp] for sp in range(C.shape[1])]
            m = np.amin(value_arr)
            _c_assignments = np.where(value_arr == m)[0]
            for k in _c_assignments:
                if j != k:
                    _t[j, k] += 1

    return _t

def sample_sankoff_path(tree, C, possible_labels=["LL", "RE", "RW", "M1", "M2", "Liv"]):
    
    def choose_assignment(vals):
        return np.random.choice(vals)
    
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    
    bfs_postorder = [root]
    for e0, e1 in nx.bfs_edges(tree, root):
        bfs_postorder.append(e1)
    
    node_to_i = dict(zip(bfs_postorder, range(len(tree.nodes))))
    label_to_j = dict(zip(possible_labels, range(len(possible_labels))))
    
    
    assignments = {}
    
    # choose a minimal cost assignment for root based on C
    pars = np.amin(C[node_to_i[root], :])
    assignments[root] = choose_assignment(np.where(C[node_to_i[root], :] == pars)[0])
    
    # sankoff top down 
    md = cmp.get_max_depth(tree, root)
    
    # bottom up approach
    d = 0
    while d <= md: 
        
        internal_nodes = cmp.cut_tree(tree, d)
        for int_node in internal_nodes:
            children = list(tree.successors(int_node))
            s = assignments[int_node]
            
            # compute optimal assignment for child
            for c in children:
                c_i = node_to_i[c]
                value_arr = [SANKOFF_SIGMA(s, sp) + C[c_i, sp] for sp in range(C.shape[1])]
                m = np.amin(value_arr)
                assignments[c] = choose_assignment(np.where(value_arr == m)[0])
        d += 1 
                    
    
    # convert assignments back to true labels
    
    for k in assignments.keys():
        assignments[k] = possible_labels[assignments[k]]
        
    nx.set_node_attributes(tree, assignments, "label")
    return tree, pars
