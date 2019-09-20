from ete3 import Tree
import networkx as nx 
import pandas as pd 

def progagate_function(tree, fn, feature_name, min_size = 0):

	for v in tree.traverse():

		n_child = len(v.get_leaves())

		if n_child >= min_size:
			v.add_feature(feature_name, fn(v))

	return tree