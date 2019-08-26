from ete3 import Tree
import networkx as nx 
import pandas as pd 
import numpy as np
from functools import reduce
from tqdm import tqdm

from collections import OrderedDict, defaultdict

import scipy.stats as scs
import cassiopeia.TreeSolver.compute_meta_purity as cmp
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

import traceback
import multiprocessing
import concurrent.futures

from . import sankoff_parsimony

def assign_labels(tree, labels):
	
	_leaves = [n for n in tree if tree.out_degree(n) == 0]
	for l in _leaves:
		tree.nodes[l]["label"] = [labels[l.name]][0]
	
	return tree


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

	for i in range(NUM_ITER):

		_tdict = instantiate_tdict(possible_labels, _list = False)

		tree2, parsimony = sankoff_parsimony.sample_sankoff_path(tree.copy(), C, possible_labels=possible_labels)
		
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

def compute_transition_matrix(tree, meta, iterations=100, title = None, save_fp = None, show=True, plot=True):
	
	tree = assign_labels(tree, meta)

	possible_labels = meta.unique()

	C = sankoff_parsimony.sankoff(tree, possible_labels=possible_labels)

	transition_matrix, se_mat = estimate_transition_probs(C, tree, possible_labels=possible_labels, NUM_ITER=iterations)
	
	labels = transition_matrix.columns
	transition_matrix = transition_matrix.values
	np.fill_diagonal(transition_matrix, np.nan)

	transition_matrix = pd.DataFrame(transition_matrix, index=labels, columns=labels)
	
	if plot:
		# plot results
		cmap = matplotlib.cm.viridis
		cmap.set_bad("white", 1.)
		
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
		
		lgout = compute_transition_matrix(lg_to_tree[n], lg_meta[meta_item], iterations = iterations, show=False, plot=False)
		consensus_mat += lgout[0] * (weights[n] / num_lgs)

		
	labels = consensus_mat.columns
	consensus_mat = consensus_mat.values
	np.fill_diagonal(consensus_mat, np.nan)

	consensus_mat = pd.DataFrame(consensus_mat, index=labels, columns=labels)

	
	return consensus_mat

def shuffle_labels(meta): 
	inds = meta.index.values
	np.random.shuffle(inds)
	meta.index = inds
	return meta

def generate_background(lg, meta, num_threads = 1, num_shuffles = 100, num_iterations=1000):

	rmeta = meta.copy()

	executor = concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), num_threads))

	random_metas = [shuffle_labels(rmeta) for i in range(num_shuffles)]

	num_meta = len(meta.unique())
	bg_mat = np.zeros((num_meta, num_meta))

	results = []

	futures = [executor.submit(compute_transition_matrix, lg, rand_meta, num_iterations, None, None, False, False) for rand_meta in random_metas]
	# concurrent.futures.wait(futures)
   
	for future in tqdm(concurrent.futures.as_completed(futures), total = len(futures)):
		results.append(future)

	for future in results:
		rtransition_matrix, se_mat = future.result()
		bg_mat += rtransition_matrix * (1.0 / num_shuffles)

	labels = bg_mat.columns
	bg_mat = bg_mat.values
	np.fill_diagonal(bg_mat, np.nan)

	bg_mat = pd.DataFrame(bg_mat, index=labels, columns=labels)

	
	return bg_mat


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

