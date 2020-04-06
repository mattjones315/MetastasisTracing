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


def compute_log2fc(gr, bg, gene, adata):
	
	# g1_filt = adata.obs.apply(lambda x: x[groupby_var] == g, axis=1)
	# bg_filt = adata.obs.apply(lambda x: x[groupby_var] == bg, axis=1)
	
	gene_ii = np.where(adata.raw.var_names == gene)[0][0]
	
	exp_g = np.mean(gr[:,gene_ii]) + 0.01
	exp_bg = np.mean(bg[:,gene_ii]) + 0.01
	
	return np.log2(exp_g / exp_bg)

def create_DE_df(counts, groupby_var, gr, bg, result, method='ttest'):
	
	g_filt = counts.obs.apply(lambda x: x[groupby_var] ==  gr, axis=1).values
	bg_filt = counts.obs.apply(lambda x: x[groupby_var] == bg, axis=1).values
	
	print(counts.X.shape, len(g_filt), g_filt[:10])
	gdata = counts.X[g_filt, :]
	bgdata = counts.X[bg_filt, :]
	
	log2fc = {}
	adj_pvalues = {}
	scores = {}

	if method == 'logreg':
		for gene, score in zip(result['names'][gr], result['scores'][gr]):
			scores[gene] = score
			log2fc[gene] = compute_log2fc(gdata, bgdata, gene, counts)
		
		de_df = pd.DataFrame.from_dict(scores, orient='index', columns=['scores'])
		de_df['gene'] = de_df.index
		de_df['log2fc'] = de_df.index.map(log2fc)
		de_df.index = range(de_df.shape[0])
		return de_df
	
	for gene, qval, fc in zip(result['names'][gr], result['pvals_adj'][gr], result['logfoldchanges'][gr]):
		#scores[gene] = score
		#log2fc[gene] = compute_log2fc(gdata, bgdata, gene, counts)
		
		adj_pvalues[gene] = qval
		log2fc[gene] = fc
		
	de_df = pd.DataFrame.from_dict(adj_pvalues, orient='index', columns=['qval'])
	de_df['gene'] = de_df.index
	de_df['log2fc'] = de_df.index.map(log2fc)
	de_df.index = range(de_df.shape[0])
	return de_df

def consolidate_genes(summs, lgfc_thresh = 1.0, num_genes = 50):
	unique_genes = []

	for summ in summs:
		fc_pass = summ[np.abs(summ['log2fc']) >= lgfc_thresh]
		unique_genes += list(fc_pass.sort_values(by=['log2fc'], ascending=False)["gene"].iloc[1:num_genes])

	unique_genes = np.unique(unique_genes)

	return unique_genes 

def bulk_by_group(cl_assign, adata, unique_genes):

	cl_assign = adata.obs["Groupby"]
	meta_levels = np.unique(cl_assign)
	kii = np.in1d(adata.var_names, unique_genes)

	num_genes = len(adata.var_names)
	de_bulk = np.zeros((len(meta_levels), len(kii[kii == True])))


	for i in tqdm(range(len(meta_levels))):
		m = meta_levels[i]
		if m == 'nan' or m == "NaN":
			continue
		cells = np.where(cl_assign == m)[0]
		subset = adata[cells,:]
		subset2 = subset[:,kii]
		gexp = np.mean(subset2.X, axis=0)
		de_bulk[i,:] = gexp

	de_bulk = pd.DataFrame(de_bulk)
	de_bulk.index = meta_levels

	gene_names = adata.var_names[kii]
	de_bulk.columns = gene_names

	return de_bulk

def filter_data_discrete(adata, meta_var, groups):

	filt = adata.obs.apply(lambda x: x[meta_var] in groups, axis=1)
	return adata[filt,:] 

def filter_data_continuous(adata, meta_var, value):

	filt = adata.obs.apply(lambda x: x[meta_var] >= value, axis=1)
	return adata[filt,:] 

def split_data_discrete(adata, meta_var, groups):

	adata.obs["Groupby"] = adata.obs.apply(lambda x: str(int(x[meta_var])) if x[meta_var] in groups else "BG", axis=1)
	adata.obs["condition"] = adata.obs.apply(lambda x: "1" if x.Groupby != "BG" else "0", axis=1)

	return adata

def split_data_continuous(adata, meta_var, value):

	adata.obs["Groupby"] = adata.obs.apply(lambda x: "High" if x[meta_var] >= value else 'BG', axis=1)
	adata.obs["condition"] = adata.obs.apply(lambda x: "1" if x.Groupby != "BG" else "0", axis=1)

	return adata




