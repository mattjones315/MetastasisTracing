import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as scs
import networkx as nx

from collections import defaultdict

import subprocess
import os

from cassiopeia.TreeSolver.simulation_tools import dataset_generation as data_gen

import utilities.metastasis_score_utils as met_utils
import cassiopeia.TreeSolver.compute_meta_purity as cmp
from cassiopeia.Anlaysis import reconstruct_states, small_parsimony


def assign_majority_vote(t, root):
    
    def majority_vote(tree, rt):
        
        children = [node for node in nx.dfs_preorder_nodes(tree, rt) if tree.out_degree(node) == 0]
        children_vals = [tree.nodes[n]["label"] for n in children]
            
        uniq_count = np.unique(children_vals, return_counts=True)
        label = uniq_count[0][np.argmax(uniq_count[1])]
        
        return label
        
    for n in nx.dfs_preorder_nodes(t, root):
        
        if t.out_degree(n) == 0:
            t.nodes[n]['label'] = t.nodes[n]['label'][0]
        
        t.nodes[n]['label'] = majority_vote(t, n)
        
    return t
    

def compute_transitions_majority_vote(t, meta):
    
    possible_labels = meta.unique()
    
    M = len(possible_labels)
    C = np.zeros((M, M))
    label_to_j = dict(zip(possible_labels, range(len(possible_labels))))
    
    root = [n for n in t if t.in_degree(n) == 0][0]
    t = small_parsimony.assign_labels(t, meta)
    
    t = cmp.set_depth(t, root)
    
    t = assign_majority_vote(t, root)
    
    # now count transitions
    for v in nx.dfs_postorder_nodes(t, source=root):
    
        v_lab = t.nodes[v]['label']
        i = label_to_j[v_lab]

        children = list(t.successors(v))
        for c in children:
            
            c_lab = t.nodes[c]['label']
            j = label_to_j[c_lab]

            C[i, j] += 1
    
    count_mat = pd.DataFrame(C)
    count_mat.columns = possible_labels
    count_mat.index = possible_labels
    return count_mat



def compute_priors(C, S, p, mean=0.01, disp=0.1, empirical = np.array([])):
    
    sp = {}
    prior_probabilities = {}
    for i in range(0, C):
        if len(empirical) > 0:
            sampled_probabilities = sorted(empirical)
        else:
            sampled_probabilities = sorted([np.random.negative_binomial(mean,disp) for _ in range(1,S+1)])
        mut_rate = p
        prior_probabilities[i] = {'0': (1-mut_rate)}
        total = np.sum(sampled_probabilities)

        sampled_probabilities = list(map(lambda x: x / (1.0 * total), sampled_probabilities))
            
        for j in range(1, S+1):
            prior_probabilities[i][str(j)] = (mut_rate)*sampled_probabilities[j-1]

    return prior_probabilities, sp 

def get_transition_stats(tree):
    
    n_transitions = 0
    transitions = defaultdict(dict)
    freqs = defaultdict(int)
    
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    for e in nx.dfs_edges(tree, source=root):
        
        p,c = e[0], e[1]
        m_p, m_c = tree.nodes[p]['meta'], tree.nodes[c]['meta']
        if m_p != m_c:
            n_transitions += 1
            if m_c not in transitions[m_p]:
                transitions[m_p][m_c] = 0
            transitions[m_p][m_c] += 1
            
        if tree.out_degree(c) == 0:
            freqs[m_c] += 1
            
    return n_transitions, transitions, freqs

def kl_divergence(a, b):
    
    kl_a = np.sum([a[i]*np.log(a[i]/b[i]) for i in range(len(a))])
    kl_b = np.sum([b[i] * np.log(b[i]/a[i]) for i in range(len(b))])
    
    return kl_a + kl_b

no_mut_rate = 0.985
number_of_states = 40
dropout = 0.17
depth = 13
number_of_characters = 40

N_clones = 500 #number of clones to simulate
max_mu = 0.3 #max rate of metastasis
min_alpha = 0.75 #min rate of doubling
t_range = [12,16] #range of time-steps
sigma = 6 #number of tumor samples
beta = 0.00 #extinction rate

# make samples
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
sample_list = [alphabet[i] for i in range(sigma)]

# set dirichlet alpha parameter here
alp_array = [1, 1, 1, 1, 1]
assert len(alp_array) == (sigma - 1)

columner1 = ['mu', 'alpha', 'N', 'n_mets', 'DynMet', "FITCHER_KL", "NAIVE_KL", 'MAJORITYVOTE_KL', 'FITCHER_SCORR', 'NAIVE_SCORR', 'MAJORITYVOTE_SCORR']
columner2 = [sample_list[i] for i in range(len(sample_list))]
masterDF = pd.DataFrame(columns=columner1+columner2)

i = 0
for j in tqdm(range(N_clones)):

    depth = np.random.choice([10, 14], p = [0.9, 0.1])

    try:
        # draw transition matrix
        tmat = pd.DataFrame(np.zeros((6,6)), index=sample_list,  columns=sample_list)

        for k in tmat.index:
            thetas = np.random.dirichlet(alp_array)
            tmat.loc[k, [j for j in tmat.columns if j != k]] = thetas

        pp, sp = compute_priors(number_of_characters, number_of_states, 1-no_mut_rate, mean=1, disp=0.1)

        tree, params = data_gen.generate_simulated_experiment_plasticisity(pp,
            [dropout]*number_of_characters,
            characters=number_of_characters,
            subsample_percentage=0.5,
            dropout=True, 
            sample_list = sample_list, 
            max_mu = max_mu, 
            min_alpha = min_alpha,
            depth = depth, 
            beta = 0,
            transition_matrix = tmat,
            )

        n_mets, mets, freqs = get_transition_stats(tree.network)
        leaves = [n for n in tree.network if tree.network.out_degree(n) == 0]
        meta = pd.DataFrame.from_dict(dict(zip([l.name for l in leaves], [tree.network.nodes[l]['meta'] for l in leaves])), orient='index', columns = ['sample'])
        norm_fitch = met_utils.compute_dynamic_metastasis_score(tree.network, meta['sample'])
        t = tree.get_network()
        t2 = t.copy() 
        
        # meta = pd.DataFrame.from_dict(dict(zip([n.name for n in leaves], [tree.network.nodes[n]['meta'] for n in leaves])), orient='index')
        est_freqs_naive = reconstruct_states.naive_fitch(t2, meta.loc[:,'sample'])
        est_freqs = reconstruct_states.fitch_count(t, meta.loc[:,'sample'], count_unique = False)
        est_freqs_mv = compute_transitions_majority_vote(t, meta.iloc[:, 0])


        metsdf = pd.DataFrame.from_dict(mets, orient='index').loc[sample_list, sample_list]
        est_freqs = est_freqs.loc[sample_list, sample_list]
        est_freqs_naive = est_freqs_naive.loc[sample_list, sample_list]
        est_freqs_mv = est_freqs_mv.loc[sample_list, sample_list]

        est_freqs = est_freqs.fillna(value = 0)
        np.fill_diagonal(est_freqs.values,0)
        # est_freqs = est_freqs / max(1, np.sum(est_freqs.values))
        est_freqs = est_freqs.apply(lambda x: x / max(1, x.sum()), axis=1)

        np.fill_diagonal(est_freqs_naive.values,0)
        est_freqs_naive = est_freqs_naive.fillna(value = 0)
        est_freqs_naive = est_freqs_naive.apply(lambda x: x/max(1, x.sum()), axis=1)
        
        np.fill_diagonal(est_freqs_mv.values,0)
        est_freqs_mv = est_freqs_mv.fillna(value = 0)
        est_freqs_mv = est_freqs_mv.apply(lambda x: x/max(1, x.sum()), axis=1)

        np.fill_diagonal(metsdf.values,0)
        metsdf = metsdf.fillna(value = 0)
        metsdf = metsdf.apply(lambda x: x / max(1, x.sum()), axis=1)        
        # metsdf = metsdf / max(1, np.sum(metsdf.values))

        x, y = [], []
        for l in range(sigma):
            for j in range(sigma):
                if l != j and not np.isnan(est_freqs.iloc[l, j]):
                    x.append(est_freqs.iloc[l, j])
                    y.append(tmat.iloc[l, j])
        
        kl_fitcher_tmat = kl_divergence(x, y)
        scorr_fitcher = scs.spearmanr(metsdf.values.ravel(), est_freqs.values.ravel())[0]

        x, y = [], []
        for l in range(sigma):
            for j in range(sigma):
                if l != j and not np.isnan(est_freqs_naive.iloc[l, j]):
                    x.append(est_freqs_naive.iloc[l, j])
                    y.append(tmat.iloc[l, j])

        kl_naive_tmat = kl_divergence(x, y)
        scorr_naive = scs.spearmanr(metsdf.values.ravel(), est_freqs_naive.values.ravel())[0]

        x, y = [], []
        for l in range(sigma):
            for j in range(sigma):
                if l != j and not np.isnan(est_freqs_mv.iloc[l, j]):
                    x.append(est_freqs_mv.iloc[l, j])
                    y.append(tmat.iloc[l, j])

        kl_mv_tmat = kl_divergence(x, y)
        scorr_mv = scs.spearmanr(metsdf.values.ravel(), est_freqs_mv.values.ravel())[0]

        masterDF.loc[i, 'mu'] = params['mu']
        masterDF.loc[i, 'alpha'] = params['alpha']
        masterDF.loc[i, 'N'] = params['N']
        masterDF.loc[i, 'n_mets'] = n_mets
        masterDF.loc[i, 'DynMet'] = norm_fitch
        masterDF.loc[i, 'FITCHER_KL'] = kl_fitcher_tmat
        masterDF.loc[i, 'NAIVE_KL'] = kl_naive_tmat
        masterDF.loc[i, 'MAJORITYVOTE_KL'] = kl_mv_tmat
        masterDF.loc[i, 'FITCHER_SCORR'] = scorr_fitcher
        masterDF.loc[i, 'NAIVE_SCORR'] = scorr_naive
        masterDF.loc[i, 'MAJORITYVOTE_SCORR'] = scorr_mv

        for k in freqs:
            masterDF.loc[i, k] = freqs[k]
        i += 1
        
    except:
        print('here')
        continue

# masterDF.to_csv("fitcher_benchmark.uniform.txt", sep='\t')
