from ete3 import Tree
import networkx as nx 
import pandas as pd 
import numpy as np
from functools import reduce
from tqdm import tqdm

from collections import OrderedDict, defaultdict

import scipy.stats as scs
import Cassiopeia.TreeSolver.compute_meta_purity as cmp
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib


def assign_labels(tree, labels):
    
    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    for l in _leaves:
        tree.nodes[l]["label"] = [labels[l.name]][0]
    
    return tree

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

def estimate_markov_chain(tree, meta, possible_labels = ["LL", "RE", "RW", "M1", "M2", "Liv"], NUM_ITER=1000):

    def instantiate_tdict(labels, _list = True):
        """
        Helper function to instantiate transition dictionaries
        """
        transition_dict = OrderedDict()
        for l in possible_labels:
            transition_dict[l] = OrderedDict()
            for l2 in possible_labels:
                if _list:
                    transition_dict[l][l2] = []
                else:
                    transition_dict[l][l2] = 0
                    
        return transition_dict

    tree = assign_labels(tree, meta)

    possible_labels = meta.unique()

    C = sankoff(tree, possible_labels=possible_labels)

    transition_dict = instantiate_tdict(possible_labels, _list = False)
    ni_dict = defaultdict(int)

    root = [n for n in tree if tree.in_degree(n) == 0][0]

    for i in tqdm(range(NUM_ITER), desc='sampling paths'):
        tree2, parsimony = sample_sankoff_path(tree.copy(), C, possible_labels = possible_labels)

        for s, t in nx.dfs_edges(tree2, source=root):
            sj, tj = tree2.nodes[s]['label'], tree2.nodes[t]['label']
            ni_dict[sj] += 1

            transition_dict[sj][tj] += 1

    print(ni_dict)
    print(transition_dict)

    for l in transition_dict.keys():
        if ni_dict[l] == 0:
            transition_dict[l][l] = 1.0
            continue

        for l2 in transition_dict[l].keys():

            transition_dict[l][l2] /= ni_dict[l]

    P = pd.DataFrame.from_dict(transition_dict, orient='index')
    P = P.loc[possible_labels, possible_labels]

    return P



def estimate_transition_probs(C, tree, possible_labels = ["LL", "RE", "RW", "M1", "M2", "Liv"], NUM_ITER=1000):
    
    def instantiate_tdict(labels, _list = True):
        """
        Helper function to instantiate transition dictionaries
        """
        transition_dict = OrderedDict()
        for l in possible_labels:
            transition_dict[l] = OrderedDict()
            for l2 in possible_labels:
                if _list:
                    transition_dict[l][l2] = []
                else:
                    transition_dict[l][l2] = 0
                    
        return transition_dict

    #instantiate dict
    transition_dict = instantiate_tdict(possible_labels)
    root = [n for n in tree if tree.in_degree(n) == 0][0]

    for i in tqdm(range(NUM_ITER), desc="sampling paths"):

        _tdict = instantiate_tdict(possible_labels, _list = False)

        tree2, parsimony = sample_sankoff_path(tree.copy(), C, possible_labels=possible_labels)
        
        for s, t in nx.dfs_edges(tree2, source=root):
            sj, tj = tree2.nodes[s]["label"], tree2.nodes[t]['label']
            if sj != tj:
                _tdict[sj][tj] += 1 / parsimony
    
        for l in possible_labels:
            for l2 in possible_labels:
                transition_dict[l][l2].append(_tdict[l][l2])

    mean_tdict = instantiate_tdict(possible_labels, _list = False)
    for l in possible_labels:
        for l2 in possible_labels:
            mean_tdict[l][l2] = np.mean(transition_dict[l][l2])
            
    se_tdict = instantiate_tdict(possible_labels, _list = False)
    for l in possible_labels:
        for l2 in possible_labels:
            se_tdict[l][l2] = scs.sem(transition_dict[l][l2])

    transition_matrix = pd.DataFrame.from_dict(mean_tdict, orient='index')
    transition_matrix = transition_matrix.loc[possible_labels, possible_labels]

    se_matrix = pd.DataFrame.from_dict(se_tdict, orient='index')
    se_matrix = se_matrix.loc[possible_labels, possible_labels]

    return transition_matrix, se_matrix 

def estimate_and_plot_transition_probs(tree, meta, iterations=100, title = None, save_fp = None, show=True):
    
    tree = assign_labels(tree, meta)

    possible_labels = meta.unique()

    C = sankoff(tree, possible_labels=possible_labels)

    transition_matrix, se_mat = estimate_transition_probs(C, tree, possible_labels=possible_labels, NUM_ITER=iterations)
    
    labels = transition_matrix.columns
    transition_matrix = transition_matrix.values
    np.fill_diagonal(transition_matrix, np.nan)
    
    # plot results
    cmap = matplotlib.cm.viridis
    cmap.set_bad("white", 1.)
    
    transition_matrix = pd.DataFrame(transition_matrix, index=labels, columns=labels)
    h = plt.figure(figsize=(10, 10))
    sns.heatmap(transition_matrix, cmap="viridis")
    plt.ylabel("sampleID")
    plt.xlabel("sampleID")
    if title:
        plt.title(title)
    else:
        plt.title("Estimated Transition Probabilities")
        
    if save_fp:
        plt.savefig(save_fp)
    elif show:
        plt.show()
    
    return transition_matrix, se_mat

def build_consensus_transition_mat(lgs, lg_meta, meta_item, iterations = 1000, ordering=None):

    weights = {}

    total_size = np.sum([len(lg.nodes()) for lg in lgs])
    num_lgs = len(lgs)

    num_meta = len(lg_meta[meta_item].unique())

    lg_to_tree = dict(zip(range(1,num_lgs+1), lgs))    

    for n in lg_to_tree.keys():
        weights[n] = len(lg_to_tree[n].nodes()) / total_size

    consensus_mat = np.zeros((num_meta, num_meta))
    for n in tqdm(lg_to_tree.keys()):
        
        lgout = estimate_and_plot_transition_probs(lg_to_tree[n], lg_meta[meta_item], iterations = iterations, show=False)
        consensus_mat += lgout[0] * (weights[n] / num_lgs)

        
    labels = consensus_mat.columns
    consensus_mat = consensus_mat.values
    np.fill_diagonal(consensus_mat, np.nan)

    consensus_mat = pd.DataFrame(consensus_mat, index=labels, columns=labels)

    
    return consensus_mat


def plot_transition_probs(consensus_mat, save_fp = None):
    # plot results
    cmap = matplotlib.cm.viridis
    cmap.set_bad("white", 1.)
    h = plt.figure(figsize=(10, 10))
    sns.heatmap(consensus_mat, mask = np.fill_diagonal(np.zeros(consensus_mat.shape), 1), cmap="viridis")
    plt.ylabel("sampleID")
    plt.xlabel("sampleID")
    plt.title("Consensus Estimated Transition Probabilities")
    
    if save_fp:
        plt.savefig(save_fp)

    else:
        plt.show()

