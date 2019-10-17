from ete3 import Tree
import networkx as nx 
import pandas as pd 
import numpy as np
from functools import reduce
import itertools

import cassiopeia.TreeSolver.compute_meta_purity as cmp
from networkx.algorithms.traversal.depth_first_search import dfs_tree

def reconcile_fitch(T):
    
    source = [n for n in T if T.in_degree(n) == 0][0]
    for e in nx.dfs_edges(T):
        
        p, c = e[0], e[1]
        ns = np.intersect1d(T.nodes[p]['label'], T.nodes[c]['label']).tolist()
        
        if len(ns) > 0 and len(ns) == len(T.nodes[p]['label']):
            T.nodes[c]['label'] = ns 
        else:
            T.nodes[c]['label'] = list(T.nodes[c]['label'])
    return T


def count_opt_solutions(T, possible_assignments, node_to_i, label_to_j):
    
    def fill_DP(v, s):

        if T.out_degree(v) == 0:
            return 1
        
        children = list(T.successors(v))
        A = np.zeros((len(children)))
        
        for i, u in zip(range(len(children)), children):
            if s not in T.nodes[u]['label']:
                A[i] = 0
                for sp in T.nodes[u]['label']:
                    if L[node_to_i[u], label_to_j[sp]] == 0:
                        L[node_to_i[u], label_to_j[sp]] = fill_DP(u, sp)
                    A[i] += L[node_to_i[u], label_to_j[sp]]
            else:
                if L[node_to_i[u], label_to_j[s]] == 0:
                    L[node_to_i[u], label_to_j[s]] = fill_DP(u, s)
                A[i] = L[node_to_i[u], label_to_j[s]]
                
        return np.prod([A[u] for u in range(len(A))])
    
    L = np.full((len(T.nodes), len(possible_assignments)), 0.0)
    
    root = [n for n in T if T.in_degree(n) == 0][0]
    
    for s in T.nodes[root]['label']:
        L[node_to_i[root], label_to_j[s]] = fill_DP(root, s)
        
    return L

def count_num_transitions(T, L, possible_labels, node_to_i, label_to_j):
    
    def fill_transition_DP(v, s, s1, s2):
        
        if T.out_degree(v) == 0:
            return 0
        
        children = list(T.successors(v))
        A = np.zeros((len(children)))
        LS = [[]] * len(children)
        
        for i, u in zip(range(len(children)), children):
            LS_u = None
            if s in T.nodes[u]['label']:
                LS[i] = [s]
            else:
                LS[i] = T.nodes[u]['label']
            
            A[i] = 0
            for sp in LS[i]:
                A[i] += C[node_to_i[u], label_to_j[sp], label_to_j[s1], label_to_j[s2]]
            
            if (s1 == s and s2 in LS[i]):
                A[i] += L[node_to_i[u], label_to_j[s2]]
            

        parts = []
        for i, u in zip(range(len(children)), children):
            prod = 1
            
            for k, up in zip(range(len(children)), children):
                fact = 0
                if up == u:
                    continue
                for sp in LS[k]:
                    fact += L[node_to_i[up], label_to_j[sp]]
                
                prod *= fact 
                
            part = A[i] * prod
            parts.append(part)

        return np.sum(parts)
    
    C = np.zeros((len(T.nodes), L.shape[1], L.shape[1], L.shape[1]))
    root = [n for n in T if T.in_degree(n) == 0][0]

    for n in nx.dfs_postorder_nodes(T, source=root):
        for s in T.nodes[n]['label']:
            for s_pair in itertools.product(possible_labels, repeat=2):
                s1, s2 = s_pair[0], s_pair[1]
                C[node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]] = fill_transition_DP(n, s, s1, s2)
                
    return C
    
def fitch_bottom_up(tree, root):
    
    md = cmp.get_max_depth(tree, root)
    
    # bottom up approach
    d = md - 1
    while d >= 0:
        
        internal_nodes = cmp.cut_tree(tree, d)
        for i in internal_nodes:
            children = list(tree.successors(i))

            if len(children) == 1:
                tree.nodes[i]["label"] = tree.nodes[children[0]]["label"]
                continue
            if len(children) == 0:
                if 'label' not in tree.nodes[i].keys():
                    raise Exception("This should have a label!")
                continue
            
            _intersect = reduce(np.intersect1d, [tree.nodes[c]["label"] for c in children])
            if len(_intersect) > 0:
                tree.nodes[i]["label"] = _intersect
            else:
                tree.nodes[i]["label"] = reduce(np.union1d, [tree.nodes[c]["label"] for c in children])
            
        d -= 1 
    
    return tree

def fitch_top_down(tree, root):
    
    md = cmp.get_max_depth(tree, root)
    
    # Phase 2: top down assignment
    tree.nodes[root]["label"] = tree.nodes[root]["label"][0]
    d = 1
    while d <= md:
        
        internal_nodes = list(cmp.cut_tree(tree, d))
        
        for i in internal_nodes:
            
            parent = list(tree.predecessors(i))[0]
            
            if tree.nodes[parent]["label"] in tree.nodes[i]["label"]:
                tree.nodes[i]["label"] = tree.nodes[parent]["label"]
                
            else:
                tree.nodes[i]["label"] = tree.nodes[i]["label"][0]
        d += 1
        
    return tree

def fitch(tree):
    """
    Runs Fitch algorihth on tree given the labels for each leaf. Returns the tree with labels on internal node.
    """
    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    
    tree = cmp.set_depth(tree, root)
    tree = fitch_bottom_up(tree, root)
    
    tree = fitch_top_down(tree, root)
    
        
    return tree

def assign_labels(tree, labels):
    
    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    for l in _leaves:
        tree.nodes[l]["label"] = [labels.loc[l.name]]
        
    return tree


def score_parsimony(tree):
    
    score = 0
    for e in tree.edges():
        source = e[0]
        dest = e[1]

        if "label" not in tree.nodes[source] or "label" not in tree.nodes[dest]:
        	raise Exception("Internal Nodes are not labeled - run fitch first")

        if tree.nodes[source]["label"] != tree.nodes[dest]["label"]:
            score += 1
    
    return score 

def score_parsimony_cell(tree, root, cell_label):

    score = 0

    path = nx.shortest_path(tree, root, cell_label)

    i = 0
    while i < len(path) - 1:

        source = path[i]
        dest = path[i+1]

        if "label" not in tree.nodes[source] or "label" not in tree.nodes[dest]:
            raise Exception("Internal Nodes are not labeled - run fitch first")

        if tree.nodes[source]["label"] != tree.nodes[dest]["label"]:
            score += 1

        i += 1

    return score 
