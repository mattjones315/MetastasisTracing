from ete3 import Tree
import networkx as nx 
import pandas as pd 
import numpy as np
from functools import reduce

import cassiopeia.TreeSolver.compute_meta_purity as cmp

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
