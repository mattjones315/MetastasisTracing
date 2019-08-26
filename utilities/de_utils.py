import numpy as np
import scanpy.api as sc
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import os
import gseapy

from ete3 import Tree

import pickle as pic

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.stats.multitest as multi

import scipy.stats as scs
from tqdm import tqdm

def run_lin_reg(adata, gene_sig, meta_var):
	
	pvals = {}
	
	meta_scores = adata.obs[meta_var]
	signif_genes = []
	for gene in gene_sig:
		if gene not in adata.var_names:
			continue
		ex = adata.X[:,(adata.var_names == gene)][:,0]
		slope, intercept, r_value, p_value, std_err = scs.linregress(ex, meta_scores)
		pvals[gene] = p_value
		
		if p_value < 0.01:
			signif_genes.append(gene)
	return pvals, signif_genes

def run_lin_reg_ALL_GENES(adata, meta_var):
	
	pvals = {}
	betas = {}
	corrs = {}
	
	meta_scores = adata.obs[meta_var]

	for gene in tqdm(adata.var_names):
		ex = adata.X[:,(adata.var_names == gene)][:,0]
		slope, intercept, r_value, p_value, std_err = scs.linregress(meta_scores, ex)
		
		pvals[gene] = p_value
		betas[gene] = slope
		corrs[gene] = r_value
		
	return pvals, betas, corrs
