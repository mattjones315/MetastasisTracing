from ete3 import Tree, NodeStyle, TreeStyle
import networkx as nx 
import pandas as pd 
import numpy as np

import scipy as sp

import cassiopeia.TreeSolver.data_pipeline as dp

from tqdm import tqdm

import matplotlib.pyplot as plt

def get_edit_distance(n1, n2):

	count = 0
	for i in range(0, len(n1)):
		if n1[i] == n2[i] or n2[i] == '-':
			count += 0
		
		elif n1[i] == '0':
			count += 1
			
		else:
			return -1
	return count

def compute_f_size(tree, fun):
	
	fsize = 0
	for v in tree.traverse('levelorder'):
		deg = len(v.children)
		fsize += fun(deg)
		
	return fsize

def compute_sackins_index(tree, cm, use_lengths = True):
	
	I = 0
	_leaves = tree.get_leaf_names()
	N = len(_leaves)
	
	for l in _leaves:
		if use_lengths:
			node = (tree&l)
			D = 0
			while node.up:
				t, s = node.name.split("_")[0], node.up.name.split("_")[0]

				if "|" in t:
					t_charstring = t.split("_")[0].split("|")
				else:
					t_charstring = cm.loc[t].values

				if "|" in s:
					s_charstring = s.split("_")[0].split("|")
				else:
					s_charstring = cm.loc[s].values

				D += get_edit_distance(s_charstring, t_charstring)

				node = node.up
		else:
			D = (tree&l).get_distance(tree)
		I += D
	
	E = 2*(sp.special.polygamma(0, N+1) + np.euler_gamma -1)
	max_s = N * (N-1)/2 + N - 1
	E2 = 2 * np.sum([1 / k for k in range(2, N)])
	Eapprox = 2 * N * np.log10(N)
	
	
	if max_s == N:
		return 0, N
	
	normI = (I - E) / (E+1)
	normI2 = (I / len(_leaves))
	normI3 = (I - Eapprox) / (N) 
	normI4 = (I - N)/(max_s - N)
	
	return normI3, N

def compute_num_cherries(tree):
	
	C = 0
	_leaves = tree.get_leaves()
	N = len(_leaves)
	
	for i in range(len(_leaves)-1):
		for j in range(i+1, len(_leaves)):
			l1, l2 = _leaves[i], _leaves[j]
			
			p1, p2 = l1.up, l2.up
			if p1 == p2:
				C += 1
				
				
	E = N / 3 
	
	normC = C / E 
	return normC, N
	

def compute_balance_index(tree, cm, method = 'sackins'):
	
	b_inds = []
	clade_sizes = []
	if method == 'sackins':
		for subtree in tqdm(tree.traverse('preorder')):
			b_ind, N = compute_sackins_index(subtree, cm)
			b_inds.append(b_ind)
			clade_sizes.append(N)
		
	if method == 'cherries':
		
		for subtree in tqdm(tree.traverse('preorder')):
			b_ind, N = compute_num_cherries(subtree)
			b_inds.append(b_ind)
			clade_sizes.append(N)
		
	return b_inds, clade_sizes

def assess_significance(ind, N, num_sim = 100):
	
	random_inds = []
	for p in tqdm(range(num_sim)):
		
		rt = Tree()
		rt.populate(N)
		
		bind, N = compute_sackins_index(rt, None, use_lengths=False)
		random_inds.append(bind)
		
	emp = len(np.where(np.array(random_inds) > ind)[0])
	return emp / num_sim

def find_significant_nodes(tree, cm, index='sackins', clade_cutoff = 20, num_sim = 1000, verbose=True):

	nnodes = 0
	for v in tree.traverse():
		nnodes += 1

	num_scored = 0
	for v in tqdm(tree.traverse(), total = nnodes):

		s, n = compute_sackins_index(v, cm, use_lengths=False)
		if n > clade_cutoff:
			p = assess_significance(s, n, num_sim=num_sim)

			v.add_features(significance = p)
			num_scored += 1

		else:
			v.add_features(significance = 1.1)

	if verbose:
		print("Scored " + str(num_scored) + " nodes.")

	return tree

def plot_tree_with_significance(tree, out_fp, signif_cutoff = 0.01):

	# set styles
	sig_nstyle = NodeStyle()
	sig_nstyle['fgcolor'] = 'darkred'

	u_nstyle = NodeStyle()
	u_nstyle['fgcolor'] = 'blue'

	cstyle = TreeStyle()
	cstyle.mode = 'c'
	cstyle.scale = 20
	cstyle.show_leaf_name = False

	for v in tree.traverse():

		try:
			p = v.significance
			if p <= 1.0:
				print(p)
		except:
			raise Exception("You need to assess significance first -- use find_significant_nodes")
		if p <= signif_cutoff:
			v.set_style(sig_nstyle)
		else:
			v.set_style(u_nstyle)

	tree.render(out_fp, tree_style = cstyle)
