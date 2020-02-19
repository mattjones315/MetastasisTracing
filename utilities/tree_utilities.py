from ete3 import Tree
import networkx as nx
import pandas as pd
import numpy as np

from tqdm import tqdm


def progagate_function(tree, fn, feature_name, min_size=0):

    for v in tree.traverse():

        n_child = len(v.get_leaves())

        if n_child >= min_size:
            v.add_feature(feature_name, fn(v))

    return tree


def propagate_function_nx(net, fn, feature_name, min_size=0):

    root = [n for n in net if net.in_degree(n) == 0][0]

    for v in tqdm(
        nx.dfs_postorder_nodes(net, source=root),
        total=len([n for n in net.nodes]),
        desc="Propagating function down tree",
    ):

        children = list(net.successors(v))
        n_child = len([n for n in children if net.out_degree(n) == 0])

        if n_child >= min_size:
            net.nodes[v][feature_name] = fn(v)
        else:
            net.nodes[v][feature_name] = None

    return net


def aggregate_expression(nodes, data, col):

    filt = list(map(lambda x: x in nodes, data.obs_names))
    if data[filt, :].shape[0] == 0:
        return None

    count = np.mean(data.X.todense()[:, (data.var_names == col)][filt, :].ravel())
    return count


def aggregate_signature(nodes, data, col):

    filt = np.intersect1d(nodes, data.index.values)

    if len(filt) == 0:
        return None

    count = np.mean(data.loc[filt, col])
    return count
